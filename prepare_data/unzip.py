import os, zipfile

folder = '/home/beomgon/pytorch/LBP_scl/PapSwinObjDet/data/'
anno = 'annotations/'
extension = ".zip"
cnt1 = 0
cnt2 = 0
for file in os.listdir(folder+anno):
    print(file)
    if file.endswith('.zip') :
        if not file.startswith('drive-') :
            cnt1 += 1
            d = file.split('-c')[0]
#             print(d)
            zip = zipfile.ZipFile(anno + file)
            if not os.path.isdir(anno + d) :
                cnt2 += 1
                print(d)
                os.mkdir(anno + d)        
                zip.extractall(path=anno + d)
        #         zipfile.ZipFile.extract(item)
            #     if item.endswith(extension):
            #         zipfile.ZipFile.extract(item)
print('cnt1 {} cnt2 {}'.format(cnt1, cnt2))            