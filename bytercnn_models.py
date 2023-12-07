import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def byte_rcnn_model(maxlen, embed_dim, RNN_SIZE, CNN_SIZE, kernels, output_cnt, OUTPUT_PATH, initial_learning_rate = 0.001):
    # do not allocate all GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # use fp16 for faster inference
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
    strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)

    with strategy.scope():
        inputs = layers.Input(shape=(maxlen,))
        emb = layers.Embedding(maxlen, embed_dim)(inputs)
        x_context = layers.Dropout(0.1)(emb)
        x_context = layers.Bidirectional(layers.GRU(RNN_SIZE, return_sequences=True))(x_context)
        x_context = layers.Bidirectional(layers.GRU(RNN_SIZE, return_sequences=True))(x_context)
        x = layers.Concatenate()([emb, x_context])

        convs = []
        for i in range(len(kernels)):
            convs.append(tf.keras.layers.Conv1D(CNN_SIZE, kernels[i], activation=layers.LeakyReLU(alpha=0.3))(x))
            convs.append(tf.keras.layers.MaxPool1D(4)(x))
            convs.append(tf.keras.layers.Conv1D(CNN_SIZE, kernels[i], activation=layers.LeakyReLU(alpha=0.3))(x))

        poolings = [layers.GlobalAveragePooling1D()(conv) for conv in convs] + [layers.GlobalMaxPool1D()(conv) for conv
                                                                                in convs]
        x = layers.Concatenate()(poolings)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(512, activation="relu")(x)
        outputs = layers.Dense(output_cnt, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        if os.path.exists(OUTPUT_PATH + "best_model_rcnn_/"):
            model = tf.keras.models.load_model(OUTPUT_PATH + "best_model_rcnn_/")

        opt = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-09,
                                       amsgrad=True)

        model.compile(
            optimizer=opt
            , loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    return model