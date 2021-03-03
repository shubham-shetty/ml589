l = [0,0.001, 0.01, 0.1,1, 10, 100]
a = []
for i in l:
    a.append([i, linear_reg_train(X_trn, y_trn, i)[0], linear_reg_train(X_trn, y_trn, i)[1], 
              linear_reg_train(X_trn, y_trn, i)[2]])
df = pd.DataFrame(np.array(a),columns=['Lambda (l)', 'W1', 'W2', 'W3'])
df.plot(x='Lambda (l)').set(ylabel="Ridge Coefficients");