const hdf5Lib = require('.');
const globals = require('./lib/globals');

file = new hdf5Lib.hdf5.File('/Volumes/Spare/Projects/Janelia/Shares/mousebrainmicro/mousebrainmicro/registration/Database/transform.h5', globals.Access.ACC_RDONLY);

count = [3, 1, 1, 1];
const v = hdf5Lib.hdf5.openDataset(file.id, 'DisplacementField', {
    count: count
});

console.log(v);

for (let idx = 0; idx < 2; idx++) {
    var data = hdf5Lib.hdf5.readDatasetHyperSlab(v.memspace, v.dataspace, v.dataset, v.rank, {
        start: [0, 3, 2, 1],
        stride: [1, 1, 1, 1],
        count: [3, 1, 1, 1]
    });
    console.log(data.data[0][0][0]);
}

hdf5Lib.hdf5.closeDataset(v.memspace, v.dataspace, v.dataset);
