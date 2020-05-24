rm -rf templates
cd html
yarn build
cd  ..
mv html/build templates
