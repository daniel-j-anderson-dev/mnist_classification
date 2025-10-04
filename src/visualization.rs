use crate::{Image, IMAGE_WIDTH};

pub fn to_ascii_art(image: impl Image) -> String {
    let index = image.index();
    to_string(image, format!("{}\n", index), |&byte| {
        String::from(if byte >= 230 { "@" } else { "." })
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
    use crate::{visualization::*, Image, TestImage, TrainingImage};
    use std::{fs::File, io::Write, path::Path};

    fn create_directory_if_doesnt_exist(path: impl AsRef<Path>) {
        if path.as_ref().is_dir() {
        } else {
            std::fs::create_dir_all(path).unwrap();
        }
    }

    #[test]
    fn training_image_ascii_art() {
        use std::{fs::File, io::Write};

        let ascii_art_image_data = TrainingImage::all()
            .flat_map(|i| [to_ascii_art(i), "\n".into()])
            .collect::<String>();

        create_directory_if_doesnt_exist("./training_images_ascii_art");

        File::options()
            .write(true)
            .truncate(true)
            .create(true)
            .open("./training_images_ascii_art/training_images_ascii_art.txt")
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

        create_directory_if_doesnt_exist("./test_images_ascii_art");

        File::options()
            .write(true)
            .truncate(true)
            .create(true)
            .open("./test_images_ascii_art/test_images_ascii_art.txt")
            .unwrap()
            .write_all(ascii_art_image_data.as_bytes())
            .unwrap();
    }

    #[test]
    fn training_images_pgm() {
        create_directory_if_doesnt_exist("./training_images_pgm");
        for image in TestImage::all() {
            let index = image.index();
            let image_path = format!("./training_images_pgm/training_image_{}.pgm", index);
            let image_data = to_pgm(image);

            File::options()
                .write(true)
                .truncate(true)
                .create(true)
                .open(image_path)
                .unwrap()
                .write_all(image_data.as_bytes())
                .unwrap();
        }
    }

    #[test]
    fn test_images_pgm() {
        create_directory_if_doesnt_exist("./test_images_pgm");

        for image in TestImage::all() {
            let image_index = image.index();
            let image_path = format!("./test_images_pgm/test_image_{}.pgm", image_index);
            let image_data = to_pgm(image);

            File::options()
                .write(true)
                .truncate(true)
                .create(true)
                .open(image_path)
                .unwrap()
                .write_all(image_data.as_bytes())
                .unwrap();
        }
    }
}
