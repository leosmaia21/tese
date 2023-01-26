from decimal import Decimal
import pandas as pd
from PIL import Image
import rasterio
import cv2


def map(value, min1, max1, min2, max2):
    return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2


def get_labels():
    filename = "../images/Arcos-lrm.tif"
    anotacoes = "anotacoes_castros.csv"
    step = 100

    geoRef = rasterio.open(filename)
    tifGeoCoord = (geoRef.bounds[0], geoRef.bounds[1], geoRef.bounds[2], geoRef.bounds[3])
    print("coordenadas reais:", tifGeoCoord)

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(filename).convert('RGB')
    width_im, height_im = image.size


    dim = 640
    slide = round(dim * step / 100)

    xmin, ymin, xmax, ymax = 0, 0, dim, dim

    rows = round((height_im / dim) / (step / 100))
    columns = round((width_im / dim) / (step / 100))

    data = pd.read_csv(anotacoes)
    df = pd.DataFrame(data, columns=['WKT', 'Id'])

    mamoas = []

    for i in range(len(data)):
        mamoas.append(df.loc[i][0].split(","))

    final = []
    i = 0
    for m in mamoas:
        for mm in m:
            d = mm.split(" ")
            d[0] = int(map(float(d[0]), tifGeoCoord[0], tifGeoCoord[2], 0, width_im))
            d[1] = int(map(float(d[1]), tifGeoCoord[1], tifGeoCoord[3], height_im, 0))
            d.append(i)
            final.append(d)
        i += 1
    
    print(final)
    for row in range(round(rows)):
        for column in range(round(columns)):
            aux = []
            for m in final:
                if xmin <= m[0] and xmax >= m[0] and ymin <= m[1] and ymax >= m[1]:
                    aux.append(m)
            if len(aux) > 0:
                img_cropped = image.crop((xmin, ymin, xmax, ymax))
                img_cropped.save("teste_castros/" + "Arcos_" + str(row) + "_" + str(column) + ".tif")
                img_cropped = cv2.imread("teste_castros/" + "Arcos_" + str(row) + "_" + str(column) + ".tif")	#type: ignore
                for a in aux:
                    img_cropped = cv2.circle(img_cropped, (int(a[0] - xmin), int(a[1] - ymin)), radius=0, color=(0, 0, 255), thickness=3)
                cv2.imwrite("teste_castros/" + "Arcos_" + str(row) + "_" + str(column) + ".tif", img_cropped)
                # aqui
                # with open("crops_/" + "Coura_" + str(row) + "_" + str(column) + ".txt", "a") as f:
                #     index = aux[0][2]
                #     s = "0 "
                #     for a in aux:
                #         if a[2] == index:
                #             s = s + str((a[0] - xmin) / dim) + " " + str((a[1] - ymin) / dim) + " "
                #         else:
                #             print(str(row) + str(column))
                #             s = s + "\n"
                #             index = a[2]
                #             s = s + "0 "
                #     f.write(s)
            xmin += slide
            xmax += slide
        xmin = 0
        xmax = dim
        ymin += slide
        ymax += slide



if __name__ == '__main__':
    get_labels()
