use crate::{Image, image::IMAGE_WIDTH};

pub fn to_ascii_art(image: impl Image) -> String {
    to_string(image, String::new(), |&b| {
        if b >= 230 { "@" } else { "." }.into()
    })
}

pub fn to_pgm(image: impl Image) -> String {
    const PGM_HEADER: &str = "P2\n28 28\n255\n";
    to_string(image, PGM_HEADER.into(), |&b| format!("{}\n", b))
}

fn to_string(
    image: impl Image,
    initial_string: String,
    byte_conversion_function: impl Copy + FnMut(&u8) -> String,
) -> String {
    image
        .as_bytes()
        .chunks(IMAGE_WIDTH)
        .flat_map(move |row| {
            row.into_iter()
                .map(byte_conversion_function)
                .chain(["\n".into()])
        })
        .filter(|s| !s.is_empty())
        .fold(initial_string, |accumulator, byte| {
            format!("{accumulator}{byte}")
        })
}

#[cfg(test)]
mod test {
    use crate::{Image, Label, TestImage, TrainingImage, TrainingLabel, visualization::*};
    use std::{fs::File, io::Write, path::Path};

    fn create_directory_if_doesnt_exist(path: impl AsRef<Path>) {
        if path.as_ref().is_dir() {
        } else {
            std::fs::create_dir_all(path).unwrap();
        }
    }

    const TRAINING_IMAGE_ASCII_ART_DIR: &str = "./training_images_ascii_art";
    const TEST_IMAGE_ASCII_ART_DIR: &str = "./test_images_ascii_art";
    const TRAINING_IMAGE_PGM_DIR: &str = "./training_images_pgm";
    const TEST_IMAGE_PGM_DIR: &str = "./test_images_pgm";

    #[test]
    fn training_image_ascii_art() {
        use std::{fs::File, io::Write};

        let ascii_art_image_data = TrainingImage::all()
            .zip(TrainingLabel::all())
            .map(|(image, label)| {
                format!(
                    "index: {}\ndata class: {:?}\n{}",
                    image.index(),
                    label.digit_class(),
                    to_ascii_art(image),
                )
            })
            .collect::<String>();

        create_directory_if_doesnt_exist(TRAINING_IMAGE_ASCII_ART_DIR);

        File::options()
            .write(true)
            .truncate(true)
            .create(true)
            .open(format!(
                "{}/training_images_ascii_art.txt",
                TRAINING_IMAGE_ASCII_ART_DIR
            ))
            .unwrap()
            .write_all(ascii_art_image_data.as_bytes())
            .unwrap();
    }

    #[test]
    fn test_images_ascii_art() {
        use std::{fs::File, io::Write};

        let ascii_art_image_data = TestImage::all()
            .flat_map(|i| [to_ascii_art(i), "\n".into()])
            .collect::<String>();

        create_directory_if_doesnt_exist(TEST_IMAGE_ASCII_ART_DIR);

        File::options()
            .write(true)
            .truncate(true)
            .create(true)
            .open(format!(
                "{}/test_images_ascii_art.txt",
                TEST_IMAGE_ASCII_ART_DIR
            ))
            .unwrap()
            .write_all(ascii_art_image_data.as_bytes())
            .unwrap();
    }

    #[test]
    fn training_images_pgm() {
        create_directory_if_doesnt_exist(TRAINING_IMAGE_PGM_DIR);
        for image in TestImage::all() {
            File::options()
                .write(true)  
                .truncate(true)
                .create(true)
                .open(format!(
                    "{}/training_image_{}.pgm",
                    TRAINING_IMAGE_PGM_DIR,
                    image.index()
                ))
                .unwrap()
                .write_all(to_pgm(image).as_bytes())
                .unwrap();
        }
    }

    #[test]
    fn test_images_pgm() {
        create_directory_if_doesnt_exist(TEST_IMAGE_PGM_DIR);
        for image in TestImage::all() {
            File::options()
                .write(true)
                .truncate(true)
                .create(true)
                .open(format!(
                    "{}/test_image_{}.pgm",
                    TEST_IMAGE_PGM_DIR,
                    image.index()
                ))
                .unwrap()
                .write_all(to_pgm(image).as_bytes())
                .unwrap();
        }
    }
}
