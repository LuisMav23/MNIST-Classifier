import cv2

def image_transform(img, gray_scale=False, save_copy=False, save_path='./', new_size=(28,28)):
    # Read the image
    # img = cv2.imread(input_image_path)
    # if img == None :
    #     print('path incorrect')
    #     return

    # Define the new dimensions (width and height)
    new_width = new_size[0]
    new_height = new_size[1]

    # Resize the image to the new dimensions using interpolation
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    if gray_scale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if save_copy == True:
        cv2.imwrite(img=img , filename=save_path)
        
    return img

