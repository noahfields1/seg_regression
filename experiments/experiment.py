def log_prediction(yhat,x,c,p,meta,path):
    cpred = pred_to_contour(yhat)
    ctrue = pred_to_contour(c)

    new_meta = {}
    for k in meta: new_meta[k] = meta[k]

    new_meta['center']   = p.tolist()
    new_meta['yhat_raw'] = yhat.tolist()
    new_meta['c_raw']    = c.tolist()

    new_meta['yhat_centered'] = cpred.tolist()
    new_meta['c_centered']    = ctrue.tolist()

    cpred_pos = cpred+p
    ctrue_pos = ctrue+p

    new_meta['yhat_pos'] = cpred_pos.tolist()
    new_meta['c_pos']    = ctrue_pos.tolist()

    name = meta['image']+'.'+meta['path_name']+'.'+str(meta['point'])

    write_json(new_meta, path+'/predictions/{}.json'.format(name))

    plt.figure()
    plt.imshow(x,cmap='gray',extent=[-1,1,1,-1])
    plt.colorbar()
    plt.scatter(cpred_pos[:,0], cpred_pos[:,1], color='r', label='predicted',s=4)
    plt.scatter(ctrue_pos[:,0], ctrue_pos[:,1], color='y', label='true', s=4)
    plt.legend()
    plt.savefig(path+'/images/{}.png'.format(name),dpi=200)
    plt.close()
