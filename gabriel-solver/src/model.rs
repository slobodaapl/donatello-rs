use std::sync::OnceLock;

use wide::f32x8;

const ARCHIVE: &[u8] = include_bytes!("../assets/shielded_ppo.pt");
const FEATURE_COUNT: usize = 64;
const HIDDEN_COUNT: usize = 128;
const ACTION_COUNT: usize = 36;

pub struct ActorModel {
    prior_scale: f32,
    prior: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    actor: Vec<f32>,
    actor_bias: Vec<f32>,
}

impl ActorModel {
    pub fn bundled() -> Result<&'static Self, String> {
        static MODEL: OnceLock<Result<ActorModel, String>> = OnceLock::new();
        MODEL.get_or_init(Self::load).as_ref().map_err(Clone::clone)
    }

    pub fn logits(&self, features: &[f32; FEATURE_COUNT]) -> [f32; ACTION_COUNT] {
        let mut hidden1 = [0.0; HIDDEN_COUNT];
        dense_tanh(&self.w1, &self.b1, features, &mut hidden1);
        let mut hidden2 = [0.0; HIDDEN_COUNT];
        dense_tanh(&self.w2, &self.b2, &hidden1, &mut hidden2);
        let mut logits = [0.0; ACTION_COUNT];
        for (row, value) in logits.iter_mut().enumerate() {
            *value = self.actor_bias[row]
                + dot(
                    &self.actor[row * HIDDEN_COUNT..(row + 1) * HIDDEN_COUNT],
                    &hidden2,
                )
                + self.prior_scale
                    * dot(
                        &self.prior[row * FEATURE_COUNT..(row + 1) * FEATURE_COUNT],
                        features,
                    );
        }
        logits
    }

    fn load() -> Result<Self, String> {
        let prior_scale = tensor("data/0", 1)?[0];
        let prior = tensor("data/1", ACTION_COUNT * FEATURE_COUNT)?;
        let w1 = tensor("data/2", HIDDEN_COUNT * FEATURE_COUNT)?;
        let b1 = tensor("data/3", HIDDEN_COUNT)?;
        let w2 = tensor("data/4", HIDDEN_COUNT * HIDDEN_COUNT)?;
        let b2 = tensor("data/5", HIDDEN_COUNT)?;
        let actor = tensor("data/6", ACTION_COUNT * HIDDEN_COUNT)?;
        let actor_bias = tensor("data/7", ACTION_COUNT)?;
        Ok(Self {
            prior_scale,
            prior,
            w1,
            b1,
            w2,
            b2,
            actor,
            actor_bias,
        })
    }
}

fn dense_tanh<const INPUT: usize, const OUTPUT: usize>(
    weights: &[f32],
    bias: &[f32],
    input: &[f32; INPUT],
    output: &mut [f32; OUTPUT],
) {
    for row in 0..OUTPUT {
        output[row] = (bias[row] + dot(&weights[row * INPUT..(row + 1) * INPUT], input)).tanh();
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len() % 8, 0);
    let mut accumulator = f32x8::ZERO;
    for (left_chunk, right_chunk) in left.chunks_exact(8).zip(right.chunks_exact(8)) {
        accumulator += f32x8::new(left_chunk.try_into().unwrap())
            * f32x8::new(right_chunk.try_into().unwrap());
    }
    accumulator.reduce_add()
}

fn tensor(name_suffix: &str, elements: usize) -> Result<Vec<f32>, String> {
    let bytes = stored_zip_entry(ARCHIVE, name_suffix)?;
    let expected = elements * size_of::<f32>();
    if bytes.len() != expected {
        return Err(format!(
            "Gabriel checkpoint tensor {name_suffix} has {} bytes; expected {expected}",
            bytes.len()
        ));
    }
    let mut values = Vec::with_capacity(elements);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        if !value.is_finite() {
            return Err(format!(
                "Gabriel checkpoint tensor {name_suffix} contains a non-finite value"
            ));
        }
        values.push(value);
    }
    Ok(values)
}

fn stored_zip_entry<'a>(archive: &'a [u8], name_suffix: &str) -> Result<&'a [u8], String> {
    const CENTRAL_SIGNATURE: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
    let mut cursor = 0;
    while cursor + 46 <= archive.len() {
        let Some(relative) = archive[cursor..]
            .windows(4)
            .position(|bytes| bytes == CENTRAL_SIGNATURE)
        else {
            break;
        };
        cursor += relative;
        let compressed_size = read_u32(archive, cursor + 20)? as usize;
        let name_length = read_u16(archive, cursor + 28)? as usize;
        let extra_length = read_u16(archive, cursor + 30)? as usize;
        let comment_length = read_u16(archive, cursor + 32)? as usize;
        let local_offset = read_u32(archive, cursor + 42)? as usize;
        let name_start = cursor + 46;
        let name_end = name_start.saturating_add(name_length);
        if name_end > archive.len() {
            return Err(String::from(
                "Gabriel checkpoint has a truncated central directory",
            ));
        }
        let name = std::str::from_utf8(&archive[name_start..name_end])
            .map_err(|_| String::from("Gabriel checkpoint contains a non-UTF-8 entry name"))?;
        if name.ends_with(name_suffix) {
            if read_u16(archive, cursor + 10)? != 0 {
                return Err(format!("Gabriel checkpoint entry {name} is compressed"));
            }
            if local_offset + 30 > archive.len()
                || archive[local_offset..local_offset + 4] != [0x50, 0x4b, 0x03, 0x04]
            {
                return Err(format!(
                    "Gabriel checkpoint entry {name} has an invalid local header"
                ));
            }
            let local_name_length = read_u16(archive, local_offset + 26)? as usize;
            let local_extra_length = read_u16(archive, local_offset + 28)? as usize;
            let data_start = local_offset + 30 + local_name_length + local_extra_length;
            let data_end = data_start.saturating_add(compressed_size);
            return archive
                .get(data_start..data_end)
                .ok_or_else(|| format!("Gabriel checkpoint entry {name} is truncated"));
        }
        cursor = name_end + extra_length + comment_length;
    }
    Err(format!(
        "Gabriel checkpoint tensor {name_suffix} is missing"
    ))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, String> {
    bytes
        .get(offset..offset + 2)
        .and_then(|value| value.try_into().ok())
        .map(u16::from_le_bytes)
        .ok_or_else(|| String::from("Gabriel checkpoint is truncated"))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    bytes
        .get(offset..offset + 4)
        .and_then(|value| value.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| String::from("Gabriel checkpoint is truncated"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_checkpoint_matches_reference_zero_feature_logits() {
        let logits = ActorModel::bundled().unwrap().logits(&[0.0; FEATURE_COUNT]);
        let expected = [
            -1.977_764_1,
            -1.882_666_3,
            -1.838_044_4,
            1.752_239,
            2.731_91,
            -3.313_623_7,
            2.712_522_7,
            -1.463_936_2,
            -2.953_700_3,
            -1.955_623_6,
            -0.259_510_43,
            -1.049_518_2,
            -1.555_264_5,
            -0.841_366_77,
            0.354_939_97,
            3.714_119,
            -0.455_112_7,
            -1.349_824_8,
            -1.540_781_4,
            -3.769_200_8,
            -0.727_015_9,
            -1.848_21,
            1.808_120_1,
            0.052_346_75,
            -2.331_738,
            -1.917_462,
            -0.028_989_434,
            0.0,
            1.224_498_4,
            -0.990_868_9,
            -0.646_813_75,
            -1.195_589_7,
            3.341_196,
            -1.068_574_7,
            1.244_303_8,
            1.892_703,
        ];
        for (index, (actual, expected)) in logits.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2e-5,
                "logit {index}: {actual} != {expected}"
            );
        }
    }
}
