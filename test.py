import tarfile
import urllib.request
import netCDF4

stream = urllib.request.urlopen('ftp://filsrv.cicsnc.org/kknapp/hursat_goes/HURSAT-goes_v01_2013180N11256_DALILA.tar.gz')

with tarfile.open(fileobj=stream, mode='r:gz') as tar:
    print('success')
