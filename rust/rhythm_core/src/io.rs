use crate::RhythmError;
use ndarray::{Array1, Array2, ArrayD, IxDyn, OwnedRepr};
use ndarray_npy::NpzReader;
use std::fs::File;
use std::io::Cursor;

pub struct NpzArrays {
    source: NpzSource,
}

enum NpzSource {
    Path(String),
    Bytes(Vec<u8>),
}

impl NpzArrays {
    pub fn open(path: &str) -> Result<Self, RhythmError> {
        Ok(Self {
            source: NpzSource::Path(path.to_string()),
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RhythmError> {
        Ok(Self {
            source: NpzSource::Bytes(bytes.to_vec()),
        })
    }

    pub fn array1(&self, name: &str) -> Result<Array1<f32>, RhythmError> {
        self.by_name(name)?
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|e| RhythmError::Io(e.to_string()))
    }

    pub fn array2(&self, name: &str) -> Result<Array2<f32>, RhythmError> {
        self.by_name(name)?
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| RhythmError::Io(e.to_string()))
    }

    fn by_name(&self, name: &str) -> Result<ArrayD<f32>, RhythmError> {
        let key = if name.ends_with(".npy") {
            name.to_string()
        } else {
            format!("{}.npy", name)
        };
        match &self.source {
            NpzSource::Path(path) => {
                let file = File::open(path).map_err(|e| RhythmError::Io(e.to_string()))?;
                let mut npz = NpzReader::new(file).map_err(|e| RhythmError::Io(e.to_string()))?;
                npz.by_name::<OwnedRepr<f32>, IxDyn>(&key)
                    .map_err(|e| RhythmError::Io(e.to_string()))
            }
            NpzSource::Bytes(data) => {
                let cursor = Cursor::new(data.clone());
                let mut npz = NpzReader::new(cursor).map_err(|e| RhythmError::Io(e.to_string()))?;
                npz.by_name::<OwnedRepr<f32>, IxDyn>(&key)
                    .map_err(|e| RhythmError::Io(e.to_string()))
            }
        }
    }
}
