# Building MALMO with support for Anaconda's python 3.6  distribution.


Install the stuff recommended in the MALMO [installation
instructions](https://github.com/Microsoft/malmo/blob/master/doc/build_linux.md).

Since we don't care for Lua and C# bindings, I removed some of the dependencies
listed in the link above.

```
sudo apt-get install build-essential git cmake cmake-qt-gui openjdk-8-jdk swig xsdcxx libxerces-c-dev doxygen xsltproc ffmpeg python-tk python-imaging-tk
```

Next we make sure we have the Boost Libraries compiled against Anaconda's
python distribution and that we are actually using them.

```
conda install -c ananconda boost=1.61.0
export BOOST_ROOT=/path/to/anaconda3
```

Before generating the build files and compiling the project, make sure you have
a `gcc` version that is not too new. I compiled on Ubuntu 17.04 using
`update-alternatives` to set `gcc 4.9` as the default compiler.

Clone the project: `https://github.com/Microsoft/malmo`.

Download xs3p.xsl from this repo and copy to ..malmo/build/Schemas & malmo/Schemas
```
export MALMO_XSD_PATH=/path/to/malmo/Schemas

cd malmo
mkdir build
cd build

cmake .. -DUSE_PYTHON_VERSIONS=3.6 \
-DINCLUDE_LUA=OFF -DINCLUDE_JAVA=OFF -DINCLUDE_TORCH=OFF -DINCLUDE_CSHARP=OFF

cmake .. -DUSE_PYTHON_VERSIONS=3.6 -DINCLUDE_LUA=OFF -DINCLUDE_JAVA=OFF -DINCLUDE_TORCH=OFF -DINCLUDE_CSHARP=OFF -DXSD_LIBRARY_RELEASE=/usr/lib/x86_64-linux-gnu/libxerces-c.so -DCMAKE_CXX_COMPILER=/usr/bin/g++-4.9 -DCMAKE_C_COMPILER=/usr/bin/gcc-4.9

make install
```
