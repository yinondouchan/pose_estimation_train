import json
import requests
import zipfile
import torch
import os
import pickle
from torch.utils import data
from torchvision import transforms
from PIL import Image
from io import BytesIO
from collections import OrderedDict
from generate_cmap_paf import generate_cmap, generate_cmap_pinpoint, generate_paf, annotations_to_connections, annotations_to_peaks

import torch


class FootOnlyConfig:
    num_parts = 6
    num_links = 6

    topology = torch.Tensor([[0, 1, 0, 1],      # left big toe-> left small toe
                             [2, 3, 1, 2],      # left small toe -> heel
                             [4, 5, 2, 0],      # heel -> left big toe
                             [6, 7, 3, 4],      # right big toe-> right small toe
                             [8, 9, 4, 5],      # right small toe -> right heel
                             [10, 11, 5, 3]])     # right heel -> right big toe

    source_to_sink_map = {0: 1, 1: 2, 2: 0, 3: 4, 4: 5, 5: 3}

    part_names =   ['left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel']

    skeleton = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]

    link_parts_map = {tuple(link): i for i, link in enumerate(skeleton)}

    output_height = 56
    output_width = 56


class BodyAndFeetConfig:
    def __init__(self, output_width=56, output_height=56):
        self.output_width = output_width
        self.output_height = output_height
        
        self.topology = self.init_topology(self.skeleton, self.num_links)

    @staticmethod
    def init_topology(skeleton, num_links):
        topology = torch.Tensor(num_links, 4)

        for i, link in enumerate(skeleton):
            topology[i][0] = 2 * i
            topology[i][1] = 2 * i + 1
            topology[i][2] = link[0]
            topology[i][3] = link[1]

        return topology

    # 6 parts belong to feet and 17 parts belong to body
    num_parts = 23
    num_links = 25

    source_to_sink_map = {0: [1, 2], 1: [2, 3], 2: 4, 3: 5, 4: 6, 5: [6, 7, 11], 6: [8, 12], 7: 9, 8: 10, 9: [], 10: [],
                          11: 12, 12: [],13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 15, 19: 15, 20: 16, 21: 16, 22: 16}

    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
                [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [17, 15], [18, 15], [19, 15], [20, 16], [21, 16],
                [22, 16]]

    link_parts_map = {tuple(link): i for i, link in enumerate(skeleton)}
    
    
class BodyOnlyConfig:
    def __init__(self, output_width=56, output_height=56):
        self.output_width = output_width
        self.output_height = output_height
        
        self.topology = self.init_topology(self.skeleton, self.num_links)

    @staticmethod
    def init_topology(skeleton, num_links):
        topology = torch.Tensor(num_links, 4)

        for i, link in enumerate(skeleton):
            topology[i][0] = 2 * i
            topology[i][1] = 2 * i + 1
            topology[i][2] = link[0]
            topology[i][3] = link[1]

        return topology

    # 6 parts belong to feet and 17 parts belong to body
    num_parts = 17
    num_links = 19

    source_to_sink_map = {0: [1, 2], 1: [2, 3], 2: 4, 3: 5, 4: 6, 5: [6, 7, 11], 6: [8, 12], 7: 9, 8: 10, 9: [], 10: [],
                          11: 12, 12: [],13: 11, 14: 12, 15: 13, 16: 14}

    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
                [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3],
                [2, 4], [3, 5], [4, 6]]

    link_parts_map = {tuple(link): i for i, link in enumerate(skeleton)}


class HandConfig:

    def __init__(self):
        self.skeleton = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
        
        self.num_parts = 21
        self.num_links = len(self.skeleton)
        
        self.source_to_sink_map = dict()
        for i in range(self.num_parts):
            self.source_to_sink_map[i] = list()
            
        for link in self.skeleton:
            source, sink = link
            self.source_to_sink_map[source].append(sink)
            
        self.topology = self.init_topology(self.skeleton, self.num_links)
        
        self.link_parts_map = {tuple(link): i for i, link in enumerate(self.skeleton)}
        
        self.output_height = 56
        self.output_width = 56
            
    @staticmethod
    def init_topology(skeleton, num_links):
        topology = torch.Tensor(num_links, 4)

        for i, link in enumerate(skeleton):
            topology[i][0] = 2 * i
            topology[i][1] = 2 * i + 1
            topology[i][2] = link[0]
            topology[i][3] = link[1]

        return topology


class HandPoseDataset(data.Dataset):
    def __init__(self, data_pkl_path, config=None, transform=None, data_transform=None):
        self.config = config if config is not None else HandConfig()
        self.transform = transform
        self.data_transform = data_transform
        
        self.stdev = 1.0
        self.window = 5 * self.stdev
        self.dataset = list(self.load_dataset(data_pkl_path).values())

    def set_stdev(self, stdev):
        self.stdev = stdev
        
    @staticmethod
    def load_dataset(data_pkl_path):
        with open(data_pkl_path, 'rb') as pkl_path:
            dataset = pickle.load(pkl_path)
            
        return dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        image = self.dataset[index]['image'][1]
        keypoints_dict = self.dataset[index]['keypoints']
                    
        keypoints = torch.Tensor(keypoints_dict['hand_pts'])
        keypoints = torch.unsqueeze(keypoints, 0)
        is_left = keypoints_dict['is_left']
        
        # decode image and convert from HWC to CHW
        image = Image.open(BytesIO(image))
        
        if self.data_transform:
            for tf in self.data_transform:
                image, keypoints = tf(image, keypoints)
        
        if self.transform:
            image = self.transform(image)
        
        peaks, counts, peak_inds = annotations_to_peaks(keypoints, self.config.num_parts)
        connections = annotations_to_connections(keypoints, peak_inds, self.config.source_to_sink_map, self.config.link_parts_map)
        cmap = generate_cmap(counts, peaks, self.config.output_height, self.config.output_width, self.stdev, self.window)
        paf = generate_paf(connections, self.config.topology, counts, peaks, self.config.output_height, self.config.output_width, self.stdev, self.window)
        
        return image, cmap, paf


class FootPoseDataset(data.Dataset):

    def __init__(self, data_path, config=None, transform=None, mode='train', source='web', device='cpu', use_cache=False, cmap_kernel_type='gaussian'):
        """
        mode: can be either 'train' for training, 'val' for validation or 'raw' for raw data
        source: either 'web' for directly obtaining images from the web or 'zip' from predefined zip files
        """
        
        self.device = device
        self.cmap_kernel_type = cmap_kernel_type
        
        self.use_cache = use_cache
        self.cache = dict()

        if config is None:
            self.config = FootOnlyConfig()
        else:
            self.config = config

        with open(data_path) as json_file:
            self.raw_data = json.load(json_file)

        self.transform = transform
        self.image_id_to_data, self.keys_list = self.preprocess_raw_data(self.raw_data)
        self.mode = mode
        self.source = source

        # standard deviation for annotation heatmaps
        self.stdev = 5.0
        self.window = 5 * self.stdev

        zf = None
        self.id_to_image = dict()

        if self.source == 'zip':
            zf = zipfile.ZipFile(os.path.join('datasets', 'body_and_foot_pose', mode + '.zip'))

            # preload images
            for image_id in self.image_id_to_data:
                image_filename = str(image_id) + '.jpg'
                with zf.open(image_filename) as file:
                    self.id_to_image[image_id] = Image.open(file)

    def set_stdev(self, stdev):
        self.stdev = stdev
        self.window = 5 * self.stdev

    def set_window(self, window):
        self.window = window
        
    def set_kernel_type(self, kernel_type):
        self.cmap_kernel_type = kernel_type

    @staticmethod
    def preprocess_raw_data(raw_data):
        """
        Preprocess raw data into a dictionary mapping an image ID to its corresponding 'images' and 'annotations' entries
        """
        image_id_to_data = OrderedDict()

        # format annotations for each image id
        for i, annotation_datum in enumerate(raw_data['annotations']):
            image_id = annotation_datum['image_id']

            # foot_pose_annotations = annotation_datum['keypoints'][-6:]

            # discard datapoints with no labels
            # if not any(foot_pose_annotations):
            #     continue

            # add image entry if needed
            image_id_to_data.setdefault(image_id, dict())

            # add foot pose annotations to image annotations. Note that an image may have more than one annotation
            image_id_to_data[image_id].setdefault('annotations', list()).append(annotation_datum)

        # get image data for each image id
        for i, image_datum in enumerate(raw_data['images']):
            image_id = image_datum['id']

            try:
                image_id_to_data[image_id]['image'] = image_datum
            except KeyError:
                continue

        keys_list = list(image_id_to_data.keys())
        return image_id_to_data, keys_list

    def __len__(self):
        return len(list(self.keys_list))

    def __getitem__(self, index):
        """
        Get input-label pair.
        Input: Torch Tensor of the image
        Label: Confidence maps and part affinity fields
        """
        img, img_keypoints = self.get_image_and_annotation_data(index)

        # convert image to RGB
        num_channels = len(img.split())

        if num_channels == 3:
            img = img.convert('RGB')
        elif num_channels == 1:
            # convert from grayscale
            rgbimg = Image.new('RGB', img.size)
            rgbimg.paste(img)
            img = rgbimg
        else:
            raise Exception('Input image channel count should be either 3 or 1')

        if self.transform:
            img = self.transform(img)

        if self.mode == 'raw':
            image_id = self.keys_list[index]
            return img, image_id
        
        try:
            cmap, paf = self.cache[index]
        except KeyError:
            peaks, counts, peak_inds = annotations_to_peaks(img_keypoints, self.config.num_parts)
            connections = annotations_to_connections(img_keypoints, peak_inds, self.config.source_to_sink_map, self.config.link_parts_map)
            if self.cmap_kernel_type == 'pinpoint':
                cmap = generate_cmap_pinpoint(counts, peaks, self.config.output_height, self.config.output_width, amplify_output=False)
            else:
                cmap = generate_cmap(counts, peaks, self.config.output_height, self.config.output_width, self.stdev, self.window, device=self.device, kernel_type=self.cmap_kernel_type)
            paf = generate_paf(connections, self.config.topology, counts, peaks, self.config.output_height, self.config.output_width, self.stdev, self.window, device=self.device)
            
            if self.use_cache:
                cmap[cmap < 0.02] = 0
                paf[paf < 0.02] = 0
                self.cache[index] = (cmap.to_sparse(), paf.to_sparse())


        return img, cmap, paf

    def get_image_and_annotation_data(self, index):
        image_id = self.keys_list[index]
        image_height = self.image_id_to_data[image_id]['image']['height']
        image_width = self.image_id_to_data[image_id]['image']['width']

        if self.source == 'web':
            image_url = self.image_id_to_data[image_id]['image']['coco_url']
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
        elif self.source == 'zip':
            img = self.id_to_image[image_id]
        else:
            raise ValueError('Unknown data source: %s' % self.source)

        # in train, keypoints include both the whole body (first 17 keypoints) and the foot keypoints (last 6 keypoints)
        # in validation, keypoints include only the foot keypoints (6 keypoints)
        # we currently take only the foot keypoints
        img_keypoints_tensor = list()
        for annotation in self.image_id_to_data[image_id]['annotations']:
            foot_keypoints = annotation['keypoints'][: self.config.num_parts * 3]
            #foot_keypoints = annotation['keypoints'][-self.config.num_parts * 3:]
            img_keypoints_tensor.append(foot_keypoints)

        img_keypoints_tensor = torch.Tensor(img_keypoints_tensor).reshape(-1, self.config.num_parts, 3)

        # normalize keypoint x and y coordinates
        img_keypoints_tensor[..., 0] /= image_width
        img_keypoints_tensor[..., 1] /= image_height

        # TODO: get annotation bounding box and normalize it
        # img_bbox = self.image_id_to_data[image_id]['annotation']['bbox']
        return img, img_keypoints_tensor

    def close():
        if self.zipfile is not None:
            self.zipfile.close()

    def slice(self, from_index, to_index):
        """ Slice dataset from a start index to an end index (not including end index) """
        image_id_to_data_sliced = dict()

        keys_list_sliced = self.keys_list[from_index: to_index]
        for image_id in keys_list_sliced:
            image_id_to_data_sliced[image_id] = self.image_id_to_data[image_id]

        self.image_id_to_data = image_id_to_data_sliced
        self.keys_list = keys_list_sliced
        
    @staticmethod
    def download_resized_images(metadata_path, output_path, output_height, output_width):
        """
        Given an online dataset download the images, resize them and save them in the output folder by their image ID
        """
        raw_dataset = FootPoseDataset(metadata_path, mode='raw', transform=transforms.Compose([
                                                                               transforms.Resize((224, 224)),
                                                                               transforms.ToTensor()
                                                                               ]))
        dataloader = data.DataLoader(raw_dataset, batch_size=1, num_workers=10)
    
        pil_transform = transforms.ToPILImage()
        count = 0
        for image, image_id in dataloader:
            image_id = str(int(image_id))
            image_channels = image.shape[1]
            image_height = image.shape[2]
            image_width = image.shape[3]
    
            image_pil = pil_transform(image.reshape((image_channels, image_height, image_width)))
    
            output_image_path = os.path.join(output_path, image_id + '.jpg')
            image_pil.save(output_image_path, "JPEG")
    
            count += 1
            if count % 100 == 0: 
                print('Saved %s images out of %s' % (count, len(dataloader)))
