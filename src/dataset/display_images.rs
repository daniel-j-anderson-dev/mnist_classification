use crate::dataset::{IMAGE_DATA_OFFSET, IMAGE_HEIGHT, IMAGE_WIDTH, TEST_IMAGES, TRAINING_IMAGES};

pub fn training_image_to_ascii_art(image_index: usize) -> String {
    image_to_ascii_art(
        TRAINING_IMAGES,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        IMAGE_DATA_OFFSET,
        image_index,
    )
}

pub fn test_image_to_ascii_art(image_index: usize) -> String {
    image_to_ascii_art(
        TEST_IMAGES,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        IMAGE_DATA_OFFSET,
        image_index,
    )
}

pub fn training_image_to_pgm(image_index: usize) -> String {
    image_to_pgm(
        TRAINING_IMAGES,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        IMAGE_DATA_OFFSET,
        image_index,
    )
}

pub fn test_image_to_pgm(image_index: usize) -> String {
    image_to_pgm(
        TEST_IMAGES,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        IMAGE_DATA_OFFSET,
        image_index,
    )
}

fn image_to_ascii_art(
    image_data: &[u8],
    image_width: usize,
    image_height: usize,
    image_data_offset: usize,
    image_index: usize,
) -> String {
    image_to_string(
        image_data,
        image_width,
        image_height,
        image_data_offset,
        image_index,
        format!("{}\n", image_index),
        |&byte| String::from(if byte >= 230 { '@' } else { '.' }),
    )
}

fn image_to_pgm(
    image_data: &[u8],
    image_width: usize,
    image_height: usize,
    image_data_offset: usize,
    image_index: usize,
) -> String {
    image_to_string(
        image_data,
        image_width,
        image_height,
        image_data_offset,
        image_index,
        String::from("P2\n28 28\n255\n"),
        |byte| format!("{} ", byte),
    )
}

fn image_to_string(
    image_data: &[u8],
    image_width: usize,
    image_height: usize,
    image_data_offset: usize,
    image_index: usize,
    initial_string: String,
    byte_conversion_function: impl Copy + FnMut(&u8) -> String,
) -> String {
    let image_size = image_height * image_width;
    let start = image_data_offset + (image_size * image_index);
    let end = start + image_size;

    let mut output = initial_string;

    for row in image_data[start..end].chunks(image_width) {
        for value in row.iter().map(byte_conversion_function) {
            output.push_str(&value);
        }
        output.push('\n');
    }

    output
}

#[cfg(test)]
mod test {
    use crate::dataset::{
        display_images::{
            test_image_to_ascii_art, test_image_to_pgm, training_image_to_ascii_art,
            training_image_to_pgm,
        },
        TEST_IMAGE_COUNT, TRAINING_IMAGE_COUNT,
    };
    use std::{fs::File, io::Write, path::Path};

    fn create_directory_if_doesnt_exist(path: impl AsRef<Path>) {
        if path.as_ref().is_dir() {
        } else {
            std::fs::create_dir_all(path).unwrap();
        }
    }

    #[test]
    fn test_training_image_to_ascii_art() {
        use std::{fs::File, io::Write};

        let mut ascii_art_image_data = String::new();
        for image_index in 0..TRAINING_IMAGE_COUNT {
            ascii_art_image_data
                .push_str(&format!("{}\n", training_image_to_ascii_art(image_index)));
        }

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
    fn test_test_image_to_ascii_art() {
        use std::{fs::File, io::Write};

        let mut ascii_art_image_data = String::new();
        for image_index in 0..TEST_IMAGE_COUNT {
            ascii_art_image_data.push_str(&format!("{}\n", test_image_to_ascii_art(image_index)));
        }

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
    fn test_training_image_to_pgm() {
        create_directory_if_doesnt_exist("./training_images_pgm");
        
        for image_index in 0..TEST_IMAGE_COUNT {
            let pgm_image_data = training_image_to_pgm(image_index);

            File::options()
                .write(true)
                .truncate(true)
                .create(true)
                .open(format!(
                    "./training_images_pgm/training_image_{}.pgm",
                    image_index
                ))
                .unwrap()
                .write_all(pgm_image_data.as_bytes())
                .unwrap();
        }
    }

    #[test]
    fn test_test_image_to_pgm() {
        create_directory_if_doesnt_exist("./test_images_pgm");

        for image_index in 0..TEST_IMAGE_COUNT {
            let pgm_image_data = test_image_to_pgm(image_index);

            File::options()
                .write(true)
                .truncate(true)
                .create(true)
                .open(format!("./test_images_pgm/test_image_{}.pgm", image_index))
                .unwrap()
                .write_all(pgm_image_data.as_bytes())
                .unwrap();
        }
    }
}
