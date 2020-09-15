# モデル設定

n_in = len(X[0])
n_hidden = 200
n_out = len(Y[0])

model = Sequential()
model .add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('sigmoid'))

model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])


# モデル学習

epochs = 1000
batch_size = 100

model.fix(X_train, Y_train, epochs=epochs, batch=batch_size)

# 予測精度の評価

loss_and_matrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)