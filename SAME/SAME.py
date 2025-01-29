import numpy as np
import cv2
import torch
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision.transforms import v2

class_names = ['Agriculture', 'Beach', 'City', 'Desert', 'Forest', 'Grassland', 
               'Highway', 'Lake', 'Mountain', 'Parking', 'Port', 'Residential', 'Water', "UNCLASSIFIED"]

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class SlidingWindowObjectDetection():
    
    def __init__(self, model, device, **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs

    def image_scaling(self, image):
        transforms = v2.Compose([
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        

        image = image / 255.0
        image_tensor = ToTensor()(image)
        img = transforms(image_tensor)


        return img


    def sliding_window(self, image, step, ws):
        print("init")
        for y in range(0, image.shape[0] - ws[1], step):

            for x in range(0, image.shape[1] - ws[0], step):

                yield (x, y, image[y:y + ws[1], x:x + ws[0]])

    def image_pyramid(self, image, scale=1.5, minSize=(28, 28)):
            yield image   
            
    def get_rois_and_locs(self, pyramid):

        rois = []
        locs = []
        for image in pyramid:

            i = 0
            for (x, y, roiOrig) in self.sliding_window(image, self.kwargs['WIN_STEP'], self.kwargs['ROI_SIZE']):

                x = int(x)
                y = int(y)
                w = int(self.kwargs['ROI_SIZE'][0])
                h = int(self.kwargs['ROI_SIZE'][1])
                
                roi = self.image_scaling(roiOrig)
                
                rois.append(roi)
                locs.append((x, y, x + w, y + h))
                
                i += 1
        return rois, locs
    
    def visualize_rois(self, rois):
        fig, axes = plt.subplots(1, len(rois), figsize=(20, 6))
        for ax, roi in zip(axes, rois):
            roi = roi.numpy()

            roi = std[:,None,None] * roi + mean[:,None,None]
            roi = np.transpose(roi,[1,2,0])
            ax.imshow(roi, cmap='gray')
            
    def get_preds(self, rois, locs):
        model_rois = np.array(rois, dtype="float32")

    
        model_rois = torch.as_tensor(model_rois)
        model_rois = model_rois.to(self.device)

        with torch.no_grad():
            outputs = self.model(model_rois)

            preds = torch.nn.functional.softmax(outputs, dim=1)
        conf ,preds = torch.max(preds,1)

        print(preds)
        print("confidence:",conf)

        conf = conf.tolist()
        preds = preds.tolist()

        conf_preds = {x: [preds[x], conf[x]] for x in range(0,len(conf))}

        labels = {}
        
        for i in conf_preds:
            (label, prob) = conf_preds[i][0],conf_preds[i][1]
            #print(label,prob)

            if prob >= self.kwargs['MIN_CONF']:

                # overclassification towards Lake tiles must been taken into account. 
                # Higher confidence towards lake tiles added as a condition.

                if label != 7 or (prob >= (self.kwargs['MIN_CONF'] * 3) and label == 7):
                    box = locs[i]
                    L = labels.get(label, [])
                    L.append((box, prob))
                    labels[label] = L
                else:
                    box = locs[i]
                    #13
                    unclassified_label = 13
                    L = labels.get(unclassified_label, [])
                    L.append((box, prob))
                    labels[unclassified_label] = L
                

        return preds, labels
    
    def apply_nms(self, labels):
        nms_labels = {}
        for label in labels.keys():
            boxes = np.array([p[0] for p in labels[label]])
            proba = np.array([p[1] for p in labels[label]])
            boxes = non_max_suppression(boxes, proba)
            nms_labels[label] = boxes.tolist()
        return nms_labels
            
    def visualize_preds(self, img, nms_labels):
        clone = img.copy()
        overlay = img.copy()
        groups = {0: 0, 1:1, 2:3, 3:1, 4:0, 5:0, 6:6, 7:5, 8:0,9:3, 10:4, 11:3, 12:5,13:7}
        colors = {0:(45,106,79),1:(237, 201, 175) ,3:(184, 134, 11) ,4:(138,43,226),5:(0, 204, 255),6:(54, 69, 79),7:(0,0,0)}
        alpha = 0.5
        fig, ax = plt.subplots(figsize=(40, 12))
        
        for label in nms_labels.keys():
            boxes = nms_labels[label]
            for (startX, startY, endX, endY) in boxes:
                cv2.rectangle(overlay, (startX, startY), (endX, endY), colors[groups[label]], -1)
                #cv2.putText(clone, str(class_names[label]), (startX, startY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

        cv2.addWeighted(overlay, alpha, clone, 1 - alpha, 0, clone)
        for label in nms_labels.keys():
            for (startX, startY, endX, endY) in boxes:

                y = startY - 10 if startY - 10 > 10 else startY + 10
        clone_rgb = cv2.cvtColor(clone, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./map.jpg", clone_rgb)
        print("file should be saved")
    
    def __call__(self, img):
        pyramid = self.image_pyramid(img, scale=self.kwargs['PYR_SCALE'], minSize=self.kwargs['ROI_SIZE'])
        rois, locs = self.get_rois_and_locs(pyramid)
        if self.kwargs['VIZ_ROIS']:
            self.visualize_rois(rois)
        preds, labels = self.get_preds(rois, locs)
        nms_labels = self.apply_nms(labels)
        
        if self.kwargs['VISUALIZE']:
            self.visualize_preds(img, nms_labels)
        
        return nms_labels



