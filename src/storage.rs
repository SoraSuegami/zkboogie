use crate::FiniteRing;
use std::io::{BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;

pub trait Storage<F: FiniteRing> {
    fn store(&mut self, key: &str, data: Vec<F>);
    fn read(&mut self, key: &str) -> Vec<F>;
}

pub struct InFileStorage<F: FiniteRing> {
    dir: std::path::PathBuf,
    _f: PhantomData<F>,
}

impl<F: FiniteRing> Storage<F> for InFileStorage<F> {
    fn store(&mut self, key: &str, data: Vec<F>) {
        let path = self.dir.join(key);
        let file = std::fs::File::create(path).expect("create failed");
        let mut writer = BufWriter::new(file);
        for d in data {
            let bytes = d.to_bytes_le();
            writer.write_all(&bytes).expect("write failed");
        }
    }
    fn read(&mut self, key: &str) -> Vec<F> {
        let path = self.dir.join(key);
        let file = std::fs::File::open(path).expect("open failed");
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        loop {
            let mut buf = vec![0u8; F::bytes_size()];
            let n = reader.read(&mut buf).expect("read failed");
            if n == 0 {
                break;
            }
            data.push(F::from_bytes_le(&buf));
        }
        data
    }
}

impl<F: FiniteRing> InFileStorage<F> {
    pub fn new(dir: std::path::PathBuf) -> Self {
        Self {
            dir,
            _f: PhantomData,
        }
    }
}
