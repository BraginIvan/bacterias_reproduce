import cv2
import numpy as np
import base64
images_to_process = range(1, 107 + 1)
# images_to_process = range(45, 107 + 1)
# show verbose 0 or 1 or 2 or 3
show = 3


# leak was shifted by 4 pixels
bias = 5
with open('pngs.txt', 'w') as pngs:

    for img_id in images_to_process:
        print('img_id', img_id)
        name = "0" * (3 - len(str(img_id))) + str(img_id)
        leak_img_512 = cv2.imread(f'leak_data/{name}.png', 0)
        # IoU optimized
        cee_img = cv2.imread(f'dice_ft_cee/{name}.png', 0)
        cee_img[cee_img > 127] = 255
        cee_img[cee_img < 127] = 0
        cee_img_512=cee_img[:, :512]

        # different models which help to fit shifted leak
        focal1_img_512 = cv2.imread(f'dice_ft_focal1/{name}.png', 0)[:, :512]
        focal2_img_512 = cv2.imread(f'dice_ft_focal2/{name}.png', 0)[:, :512]
        focal3_img_512 = cv2.imread(f'dice_ft_focal3/{name}.png', 0)[:, :512]
        focal4_img_512 = cv2.imread(f'dice_ft_focal4/{name}.png', 0)[:, :512]
        # if show > 0:
        #     cv2.imshow('leak_img_512', leak_img_512)
        #     cv2.imshow('cee_img', cee_img)
        #     cv2.imshow('focal1_img_512', focal1_img_512)
        #     cv2.imshow('focal2_img_512', focal2_img_512)
        #     cv2.imshow('focal3_img_512', focal3_img_512)
        #     cv2.imshow('focal4_img_512', focal4_img_512)
        #     cv2.waitKey(0)
        best_fit_512 = np.zeros((512, 512))

        contours, hierarchy = cv2.findContours(leak_img_512, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            counter = contours[i]
            counter = counter.reshape((-1, 2))
            left_x, top_y = np.min(counter, axis=0)
            right_x, bottom_y = np.max(counter, axis=0)

            leak_counter = np.zeros((512, 512))
            hullIndex = cv2.convexHull(counter, returnPoints=False)
            cv2.fillConvexPoly(leak_counter, counter[hullIndex], (255, 255, 255))

            # we want to get pixels from predict images where leak counter is presented
            # counter can be shifted so we want to dilate it
            leak_counter_dilated = cv2.dilate(leak_counter, kernel=np.ones((3, 3)), iterations=3)
            leak_counter_clear = leak_img_512.copy()
            leak_counter_clear[leak_counter == 0] =0

            focal1_img_512_counter = focal1_img_512.copy()
            focal2_img_512_counter = focal2_img_512.copy()
            focal3_img_512_counter = focal3_img_512.copy()
            focal4_img_512_counter = focal4_img_512.copy()
            focal1_img_512_counter[leak_counter_dilated == 0] = 0
            focal2_img_512_counter[leak_counter_dilated == 0] = 0
            focal3_img_512_counter[leak_counter_dilated == 0] = 0
            focal4_img_512_counter[leak_counter_dilated == 0] = 0

            best_fit = np.zeros((512, 512))
            err_max = 512 * 512


            padded_image = np.zeros((512 + bias*2, 512 + bias*2))
            padded_image[bias:bias + 512, bias:bias + 512] = leak_counter_clear.copy()

            #lets choose which model provides the contour size close to leak

            for model_n, img_512_counter in enumerate([focal1_img_512_counter, focal2_img_512_counter, focal3_img_512_counter, focal4_img_512_counter]):
                pixels_count_gt = len(leak_counter_clear[leak_counter_clear > 127])
                pixels_count_pred = len(img_512_counter[img_512_counter > 127])
                if min(pixels_count_gt, pixels_count_pred) / max(pixels_count_gt, pixels_count_pred) > 0.3:
                    break
            # print(model_n)
            # if show > 1:
            #     cv2.imshow('leak_counter', leak_counter)
            #     cv2.imshow('leak_counter_clear', leak_counter_clear)
            #     cv2.imshow('leak_counter_dilated', leak_counter_dilated)
            #     cv2.imshow('focal1_img_512_counter', focal1_img_512_counter)
            #     cv2.imshow('focal2_img_512_counter', focal2_img_512_counter)
            #     cv2.imshow('focal3_img_512_counter', focal3_img_512_counter)
            #     cv2.imshow('focal4_img_512_counter', focal4_img_512_counter)
            #     cv2.imshow('img_512_counter', img_512_counter)
            #     cv2.waitKey(0)


            # if the leak in the edge, part of it can be removed, we want to try to pad it back
            if left_x < 3 or top_y < 3 or right_x > 512 - 3 or bottom_y > 512 - 3:
                pads = 4
            else:
                pads = 1

            pairs = []
            for pad in range(pads):
                pairs = []
                if pad > 0:
                    if left_x < 3:
                        padded_image[:, left_x + bias - pad] = padded_image[:, left_x + bias]
                    elif top_y < 3:
                        padded_image[top_y + bias - pad] = padded_image[top_y + bias]
                    elif bottom_y > 512 - 3:
                        padded_image[bottom_y + bias + pad - 1] = padded_image[bottom_y + bias - 1]
                    elif right_x > 512 - 3:
                        padded_image[:, right_x + bias + pad - 1] = padded_image[:, right_x + bias- 1]
                    # uncomment to see what it does
                    # cv2.imshow('padded_image', padded_image)
                    # cv2.waitKey(0)

                for x in range(bias*2):
                    for y in range(bias*2):
                        padded_image_crop = padded_image[y:y + 512, x:x + 512]
                        if np.sum(padded_image_crop) < np.sum(padded_image):
                            # if we crop white pixels it is not nesessary to continue
                            continue
                        err = np.sum(np.abs(padded_image_crop - img_512_counter)) / 255
                        if err < err_max:
                            err_max = err
                            best_fit = padded_image_crop.copy()
                            pairs = [(x, y)]
                        elif err == err_max:
                            pairs.append((x, y))

                # if different shifts produce same error we want to find the median position of these shifts
                if len(pairs) > 1:
                    print('len(pairs)', len(pairs))
                    if (5, 5) in pairs:
                        best_x = 5
                        best_y = 5
                    else:
                        pairs = np.array(pairs)
                        ys_x0 = pairs[:, 1][pairs[:, 0] == 5]
                        xs_y0 = pairs[:, 0][pairs[:, 1] == 5]
                        if len(ys_x0) != 0:
                            best_x = 5
                            best_y = ys_x0[len(ys_x0) // 2]
                        elif len(xs_y0) != 0:
                            best_y = 5
                            best_x = xs_y0[len(xs_y0) // 2]
                        else:
                            best_x, best_y = pairs[len(pairs) // 2]

                    best_fit = padded_image[best_y:best_y + 512, best_x:best_x + 512].copy()
                    err = np.sum(np.abs(best_fit - img_512_counter)) / 255
                    if err > err_max:
                        # error should not be changed
                        print(i, 'best', err, err_max)

            best_fit_512[best_fit > 0] = 255




        # separate merged contours v2 start
        fn = np.clip(best_fit_512 - cee_img_512, 0, 255)
        fp = np.clip(cee_img_512 - best_fit_512, 0, 255)
        diff = (np.abs(cee_img_512 / 255 - best_fit_512 / 255) * 255).astype('uint8')
        diff_dilated=cv2.dilate(diff, kernel=np.ones((2,2)))
        contours, _ = cv2.findContours(diff_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        for i in range(len(contours)):
            c = contours[i]
            c = c.reshape((-1, 2))
            left_x, top_y = np.min(counter, axis=0)
            right_x, bottom_y = np.max(counter, axis=0)
            contour_shape = (right_x - left_x) * (bottom_y - top_y)
            if contour_shape > 1000:
                continue
            tmp = np.zeros((512, 512))
            hullIndex = cv2.convexHull(c, returnPoints=False)
            cv2.fillConvexPoly(tmp, c[hullIndex], (255, 255, 255))
            tmp[diff==0]=0
            if np.sum(tmp / 255) > 50:
                # fne = cv2.erode(fn, np.ones((3,3)), iterations=1)
                # fpe = cv2.erode(fp, np.ones((3,3)), iterations=1)
                fne = fn.copy()
                fpe = fp.copy()
                fne[tmp == 0]=0
                fpe[tmp == 0]=0

                fnsum = np.sum(fne / 255)
                fpsum = np.sum(fpe / 255)
                if fnsum+fpsum==0:
                    continue


                if min(fnsum, fpsum) / max(fnsum, fpsum) > 0.35:
                    print('coeff', min(fnsum, fpsum) / max(fnsum, fpsum), contour_shape)
                    # cv2.imshow('fne', fne)
                    # cv2.imshow('fpe', fpe)
                    # cv2.imshow('tmp', tmp)
                    best_fit_512[tmp > 0] = cee_img_512[tmp > 0]
        # v2 finish

        # if edge is lost
        if np.sum(best_fit_512[:, 511][best_fit_512[:, 510] > 0]) == 0 and np.sum(best_fit_512[:, 510]) > 0:
            line = cv2.erode(best_fit_512[:, 510], np.ones((5, 1)), iterations=1)
            line = cv2.dilate(line, np.ones((5)), iterations=1)
            best_fit_512[:, 511] = line[:, 0]
        if np.sum(best_fit_512[:, 0][best_fit_512[:, 1] > 0]) == 0 and np.sum(best_fit_512[:, 1]) > 0:
            line = cv2.erode(best_fit_512[:, 1], np.ones((5, 1)), iterations=1)
            line = cv2.dilate(line, np.ones((5)), iterations=1)
            best_fit_512[:, 0] = line[:, 0]
        if np.sum(best_fit_512[0][best_fit_512[1] > 0]) == 0 and np.sum(best_fit_512[1]) > 0:
            line = cv2.erode(best_fit_512[1], np.ones((5)), iterations=1)
            line = cv2.dilate(line, np.ones((5, 1)), iterations=1)
            best_fit_512[0] = line[:, 0]
        if np.sum(best_fit_512[511][best_fit_512[510] > 0]) == 0 and np.sum(best_fit_512[510]) > 0:
            line = cv2.erode(best_fit_512[510], np.ones((5)), iterations=1)
            line = cv2.dilate(line, np.ones((5)), iterations=1)
            best_fit_512[511] = line[:, 0]


        # set pixels which are not in leak
        best_fit = np.zeros((512,640), dtype='uint8')
        best_fit[:,:512]=best_fit_512
        best_fit[:,512:]=cee_img[:,512:]
        best_fit[best_fit>127] = 255
        best_fit[best_fit<=127] = 0

        cv2.imwrite(f'tmp.png', best_fit)
        filename = f'tmp.png'
        with open(filename, 'rb') as fp:
            encoded_string = base64.b64encode(fp.read()).decode('utf-8')
            pngs.write(str(encoded_string) + "\n")

        # testing
        cv2.imshow('best_fit', best_fit)
        cv2.imshow('leak_img_512', leak_img_512)
        old_predict = cv2.imread(f'/home/ivan/projects/its/test_labels/{name}.png', 0)
        orig = cv2.imread(f'/home/ivan/projects/its/dataset_its/test/{name}.png')

        orig2 = orig.copy()
        orig3 = orig.copy()
        orig4 = orig.copy()

        tmp = orig2[:, :, 1]
        tmp[old_predict == 255] = 125
        orig2[:, :, 1] = tmp
        cv2.imshow('old', orig2)

        tmp = orig3[:, :, 1]
        tmp[best_fit == 255] = 125
        orig3[:, :, 1] = tmp
        cv2.imshow('new', orig3)

        delta = (np.abs(best_fit / 255 - old_predict / 255)*255).astype('uint8')
        print(np.sum(delta))
        tmp = orig4[:, :, 1]
        tmp[delta == 255] = 125
        orig4[:, :, 1] = tmp
        cv2.imshow('old_new_diff', orig4)

        cv2.imshow('delta', delta)

        cv2.waitKey(1)




