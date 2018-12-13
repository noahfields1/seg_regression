from components.common import BasePredictor
import modules.vessel_regression as vessel_regression

def log_prediction(yhat,x,c,p,meta,path):
    cpred = pred_to_contour(yhat)
    ctrue = pred_to_contour(c)
    scale  = meta['dimensions']*meta['spacing']/2

    new_meta = {}
    for k in meta: new_meta[k] = meta[k]

    new_meta['center']   = p.tolist()
    new_meta['yhat_raw'] = yhat.tolist()
    new_meta['c_raw']    = c.tolist()

    new_meta['yhat_centered_unscaled'] = cpred.tolist()
    new_meta['c_centered_unscaled']    = ctrue.tolist()

    cpred_pos = (cpred+p)*scale
    ctrue_pos = (ctrue+p)*scale

    new_meta['yhat_pos'] = cpred_pos.tolist()
    new_meta['c_pos']    = ctrue_pos.tolist()

    new_meta['radius_pixels'] = meta['radius']
    new_meta['radius'] = meta['radius']*meta['spacing']

    name = meta['image']+'.'+meta['path_name']+'.'+str(meta['point'])

    write_json(new_meta, path+'/predictions/{}.json'.format(name))

    plt.figure()
    plt.imshow(x,cmap='gray',extent=[-scale,scale,scale,-scale])
    plt.colorbar()
    plt.scatter(cpred_pos[:,0], cpred_pos[:,1], color='r', label='predicted',s=4)
    plt.scatter(ctrue_pos[:,0], ctrue_pos[:,1], color='y', label='true', s=4)
    plt.legend()
    plt.savefig(path+'/images/{}.png'.format(name),dpi=200)
    plt.close()

class EdgeFittingPredictor(BasePredictor):
    def predict(self):
        predictions = self.model.predict(self.X)

        path = self.config['RESULTS_DIR']+'/'+self.config['NAME']
        if self.data_key == "VAL":
            path = path+'/val'
        elif self.data_key == "TEST":
            path = path+'/test'

        for i in tqdm(range(predictions.shape[0])):
            x = self.X[i]
            c = self.C[i]
            p = self.points[i]
            meta = self.meta[i]
            yhat = predictions[i]

            log_prediction(yhat,x,c,p,meta,path)
