from mmseg.apis import inference_model, init_model, show_result_pyplot

def main():
    config = 'configs/model/uda_daformer_HHHead_gta2cityscapes512.py'
    checkpoint = 'workdir/best_69.33/best_mIoU_iter_51000.pth'
    img = 'data/cityscapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png'

    model = init_model(config, checkpoint, device='cuda:0')
    result = inference_model(model, img)
    show_result_pyplot(
        model,
        img,
        result,
        title='result',
        opacity=0.5,
        with_labels=False,
        draw_gt=False,
        show=False ,
        out_file='')

if __name__ == '__main__':
    main()