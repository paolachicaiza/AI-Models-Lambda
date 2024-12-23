from keras.models import Sequential
from keras import regularizers
from keras.layers import (
    BatchNormalization, Dense, Dropout
)


def model_mlp(x_train, number_landing_pages):
    try:
        model = Sequential()
        model.add(Dense(20, input_dim=x_train.shape[1],
                        kernel_initializer='he_uniform',
                        activation='relu', activity_regularizer=regularizers.l1(1e-4)))
        model.add(BatchNormalization())
        model.add(Dense(10, kernel_initializer='he_uniform',
                        activation='relu', activity_regularizer=regularizers.l1(1e-4)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(5, kernel_initializer='he_uniform',
                        activation='relu', activity_regularizer=regularizers.l1(1e-4)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(number_landing_pages, kernel_initializer='he_uniform',
                        activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        model.summary()

        return model
    except Exception as error:
        print(error)
        print('Error')
        raise error
