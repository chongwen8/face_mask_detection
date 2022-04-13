import os
import glob
import cv2
import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Input, MaxPool2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import splitfolders
from utils import data_preprocess

if __name__ == "__main__":
    INIT_LR = 1e-4
    EPOCHS = 20
    BS = 32

    dirs = os.listdir('./data')
    dirs = dirs[1:]
    data = {'path': [], 'label':[], 'name' :[]}
    for folder in dirs:
        path = glob.glob('./data/{}/*.jpg'.format(folder))
        path.extend(glob.glob('./data/{}/*.png'.format(folder)))
        name = [p.split('/')[-1] for p in path]
        label = ['{}'.format(folder)]*len(path)
        data['path'].extend(path)
    # 对应的路径对对应的标签
        data['label'].extend(label)
        data['name'].extend(name)

    data_df = pd.DataFrame(data)

    images_directory = './preprocessed_data'
    drop_rows = []
    for i in tqdm.tqdm(range(len(data_df)), desc='preprocessing'):
        # Get The File Path and Read The Image
        image = cv2.imread(data_df['path'].iloc[i])
        image_path = os.path.join(os.path.join(images_directory, data_df['label'].iloc[i]), data_df['name'].iloc[i])
        # Set The Cropped Image File Name
        data_df['path'].iloc[i] = image_path
        try:
            preprocessed_image = data_preprocess(image_path)
        except:
            print(data_df['path'].iloc[i])
        if preprocessed_image is None:
            drop_rows.append(i)
        else:
            cv2.imwrite(image_path, preprocessed_image)
    data_df.drop(drop_rows)

    splitfolders.ratio('preprocessed_data_new/', output="dataset_split_new", seed=113, ratio=(.7, .2, .1), group_prefix=None)

    main_dir = './dataset_split'
    train_dir = os.path.join(main_dir,'train')
    test_dir = os.path.join(main_dir,'test')
    valid_dir = os.path.join(main_dir,'val')

    train_datagen = ImageDataGenerator(rescale=1./255,
                                    zoom_range = 0.2,
                                    rotation_range = 40,
                                    horizontal_flip = True,
                                    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    image_target_size = (224,224)
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=image_target_size,
                                                        batch_size = BS,
                                                        seed = 113,
                                                        class_mode = 'categorical'
                                                        )
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                        target_size=image_target_size,
                                                        batch_size = BS,
                                                        seed = 113,
                                                        class_mode = 'categorical'
                                                        )
    valid_generator = validation_datagen.flow_from_directory(valid_dir,
                                                        target_size=image_target_size,
                                                        batch_size = BS,
                                                        seed = 113,
                                                        class_mode = 'categorical',
                                                        )


    baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    # model 2 CNN model
    # model = Sequential([
    #     Conv2D(16,3,padding='same',input_shape=(300,300,3),activation='relu'),
    #     MaxPool2D(),
    #     Conv2D(32,3,padding='same',activation='relu'),
    #     MaxPool2D(),
    #     Conv2D(64,3,padding='same',activation='relu'),
    #     MaxPool2D(),
    #     Flatten(),
    #     Dense(498,activation='relu'),
    #     Dense(38,activation='relu'),
    #     Dropout(0.5),
    #     Dense(3,activation='softmax'),
    # ])

    model.compile(Adam(learning_rate=0.001),
                loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])

    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])


    with tf.device("/gpu:0"):
        history = model.fit_generator(train_generator, epochs = EPOCHS, steps_per_epoch = len(train_generator), 
                                    validation_data = valid_generator, validation_steps = len(valid_generator))

    history_df = pd.DataFrame(history.history)

    history_df[['loss','val_loss']].plot(kind='line')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    history_df[['accuracy','val_accuracy']].plot(kind='line')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    test_loss , test_acc = model.evaluate(test_generator)
    print('test acc :{} test loss:{}'.format(test_acc,test_loss))

    model.save('model_X.h5')
