from astropy.io import fits
from astropy.io.fits import hdu
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
from astropy import units as u
file = "./src/data/frame-r-002505-3-0038.fits"
# fits.info(file)
# image_data = fits.getdata(file, ext=1)
# print(image_data.names)
# # print(image_data.shape)

# with fits.open("./src/data/gz1.fits") as hdu_list:
#     primary = hdu_list[1]
#     print(primary.info())587-72717-8986356823

# co = SkyCoord(ra="179.04298400878906", dec="60.522518157958984", unit="deg")
co = SkyCoord('0h8m05.63s +14d50m23.3s')
result = SDSS.query_region(co)
print(result[:5])
# xid = SDSS.query_region(co, spectro=True)

# xid = SDSS.query_region(co)
print(xid)
hdu_list = SDSS.get_images(matches=co)
primary = hdu_list[1][0].data
# primary = hdu_list[0][0]
# print(primary.data.shape)

import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
plt.figure()
plt.imshow(primary.data)
plt.colorbar()
plt.savefig("test.jpg")