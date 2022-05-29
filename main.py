from moviepy.editor import VideoFileClip
from svm_pipeline import *
from lane import *



def pipeline_svm(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_svm(img_undist, img_lane_augmented, lane_info)

    return output


if __name__ == "__main__":

    demo = 2  # 1:image (SVM), 2:video (SVM pipeline) , 3: debuging mode

    if demo == 1:

        filename = 'examples/test4.jpg'

        image = mpimg.imread(filename)


        #(2) SVM pipeline
        draw_img = pipeline_svm(image)
        fig = plt.figure()
        plt.imshow(draw_img)
        plt.title('svm pipeline', fontsize=30)
        plt.show()


    elif demo == 2:
        # SVM pipeline

        video_output = 'examples/project_output.mp4'
        clip1 = VideoFileClip("examples/project_video.mp4").subclip(2,40)
        clip = clip1.fl_image(pipeline_svm)
        clip.write_videofile(video_output, audio=False)

    else:
        rows = 2
        columns = 2

        hogCar = 'examples/31.png'
        hogNotCar = 'examples/2.png'


        hogCarImg = mpimg.imread(hogCar)
        hogNotCarImg = mpimg.imread(hogNotCar)

        grayCar = cv2.cvtColor(hogCarImg, cv2.COLOR_RGB2GRAY)
        grayNotCar = cv2.cvtColor(hogNotCarImg, cv2.COLOR_RGB2GRAY)
        feature1, hogFeatCar = get_hog_features(grayCar, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        feature2, hogFeatNotCar = get_hog_features(grayNotCar, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)


        #hog features output
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(hogFeatCar)
        fig.add_subplot(rows, columns, 2)
        plt.imshow(hogCarImg)

        plt.axis('off')
        plt.title("hog_car")
        fig.add_subplot(rows, columns, 3)
        plt.imshow(hogFeatNotCar)
        fig.add_subplot(rows, columns, 4)
        plt.imshow(hogNotCarImg)

        plt.axis('off')
        plt.title("hog_noncar")

        plt.show()
