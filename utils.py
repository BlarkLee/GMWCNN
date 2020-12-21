import numpy as np
import cv2

def computeAccuracy( pred, labelIndexBatch, maskBatch, numClasses = 21 ):
    hist = np.zeros(numClasses * numClasses, dtype=np.int64 )

    pred = pred.cpu().data.numpy()
    assert( pred.shape[1] == numClasses )
    pred = pred.argmin(axis = 1 )

    gt = labelIndexBatch.cpu().data.numpy()
    gt = gt.squeeze(1)
    mask = maskBatch.cpu().data.numpy()
    mask = mask.squeeze(1)

    assert(gt.max() < numClasses )
    assert(pred.max() < numClasses )

    sumim = gt * numClasses + pred
    sumim = sumim[mask != 0].squeeze()

    # histIm, _ = np.histogram(sumim, np.arange(numClasses * numClasses ) )
    histIm, _ = np.histogram(sumim, np.arange(numClasses * numClasses + 1 ) )
    hist[0:len(histIm ) ] += histIm

    return hist.reshape(numClasses, numClasses )

def save_label(label, mask, cmap, name, nrows, ncols ):
    label = label.cpu().numpy()
    assert(label.shape[1] == cmap.shape[0] )
    label = label.argmax(axis= 1 )

    mask = mask.cpu().numpy().squeeze(1)

    imHeight, imWidth = label.shape[1], label.shape[2]
    outputImage = np.zeros( (imHeight*nrows, imWidth*ncols, 3), dtype=np.float32 )
    for r in range(0, nrows ):
        for c in range(0, ncols ):
            imId = r * ncols + c
            if imId >= label.shape[0]:
                break

            maskIm = mask[imId, :, :][:, :, np.newaxis ]

            labelIm = label[imId, :, : ]
            labelIm = cmap[labelIm.flatten(), :]
            labelIm = labelIm.reshape(imHeight, imWidth, 3 )

            labelIm = labelIm + (1 - maskIm )

            rs = r * imHeight
            cs = c * imWidth
            outputImage[rs:rs+imHeight, cs:cs+imWidth, :] = labelIm

    outputImage = (np.clip(outputImage, 0, 1) * 255).astype(np.float32 )
    if name is None:
        return outputImage[:, :, ::-1]
    else:
        cv2.imwrite(name, outputImage[:, :, ::-1] )

    return
