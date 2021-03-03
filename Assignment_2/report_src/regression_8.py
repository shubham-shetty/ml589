# Training Regression Stump
def reg_stump_train(X_trn, y_trn):
    
    # Algorithm - 
    # 1. For each dimension, sort the data.
    # 2. Create potential split points as average of adjacent elements in sorted data.
    # 3. Define regions with points on either side of split (R1 if x_i < s else R2).
    # 4. Find optimal constants c1 & c2
    # 5. Calculate error for each split.
    # 6. Return split values for which minimum error is obtained across the dimensions.
    
    # Get number of dimensions in input
    D = X_trn.shape[1]
    
    # Initialise error as infinity
    err = float("inf")
    
    for j in range(D):
        s = []
        x = X_trn[:,j]
        z = np.sort(x)
        
        # Get potential splits
        for i in range(len(z)-1):
            s.append(0.5*(z[i]+z[i+1]))    
        
        for split in s:
            y1, y2 = 0, 0
            r1, r2 = [], []
            # Divide all points into two regions
            for xn in range(len(x)):
                if x[xn] < split:
                    r1.append(xn)
                    y1 += y_trn[xn]
                else:
                    r2.append(xn)
                    y2 += y_trn[xn]
            
            # Get constant values for both regions
            c1 = y1/len(r1)
            c2 = y2/len(r2)
            
            # Calculate error 
            err2 = sum([(y_trn[i] - c1)**2 for i in r1]) + sum([(y_trn[i] - c2)**2 for i in r2])
            
            # Replace terms if error is minimum
            if(err2<err):
                err = err2
                dim = j
                thresh = split
                c_left = c1
                c_right = c2
    
    return dim, thresh, c_left, c_right