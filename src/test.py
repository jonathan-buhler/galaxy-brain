from astropy import coordinates as coords
from astroquery.sdss import SDSS
co = coords.SkyCoord('0h8m05.63s +14d50m23.3s')
# co = coords.SkyCoord(ra="179.04298400878906", dec="60.522518157958984", unit="deg")
result = SDSS.query_region(co)
# img = SDSS.get_images(matches=result)[0][0]
img = SDSS.get_images(coordinates=co)[0][0]

data = img.data
print(data.shape)

import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
plt.figure()
plt.imshow(data)
plt.colorbar()
plt.savefig("test.jpg")
