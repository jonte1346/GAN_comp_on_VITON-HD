# GAN Comparison on Clothing Dataset

This project implements and compares multiple Generative Adversarial Networks (GANs) on a clothing image dataset. The goal is to explore the strengths and weaknesses of each architecture using both qualitative outputs and quantitative metrics like FID, KID, and Inception Score.

## Models Implemented

- **DCGAN**  
  A baseline GAN with ConvTranspose2d layers and batch normalization.

- **WGAN / WGAN-GP**  
  Uses Wasserstein loss with optional gradient penalty and spectral normalization for improved training stability.

- **InfoGAN**  
  Introduces latent code `c` to learn disentangled and interpretable representations via mutual information maximization.

- **VAE-GAN**  
  Combines a Variational Autoencoder with a GAN discriminator to improve sample sharpness and latent encoding quality.

- **Lite StyleGAN**  
  A simplified StyleGAN with Adaptive Instance Normalization (AdaIN) to inject style information into intermediate layers.

## Evaluation Metrics

- **FID (Fréchet Inception Distance)**  
  Measures similarity between real and generated image distributions.

- **KID (Kernel Inception Distance)**  
  Like FID, but unbiased and more stable on smaller datasets.

- **Inception Score (IS)**  
  Measures image quality and diversity using a pre-trained classifier.

## Dataset

- Dataset: Clothing images, preprocessed to **64×64** resolution.
- Real samples stored in: `real_clothing/real/`
- Each GAN saves generated samples in: `generated_<model_name>/`

## How to Run

 **Train a model**
   ```python
   train_dcgan(generator, discriminator, ...)
   ```

 **Evaluate**
   ```python
   report_all_fids()
   report_all_inception_scores()
   report_all_kid_flat()
   ```

## 📈 Results Summary

| Model     | FID ↓   | KID ↓     | Inception Score ↑ |
|-----------|---------|-----------|--------------------|
| DCGAN     | 302.70  | 0.3296    | 1.06 ± 0.00        |
| WGAN      | 262.95  | 0.2705    | 1.06 ± 0.00        |
| InfoGAN   | 330.87  | 0.3343    | 1.05 ± 0.00        |
| VAE-GAN   | 269.29  | 0.2924    | 1.06 ± 0.00        |
| StyleGAN  | 300.15  | 0.2864    | 1.07 ± 0.00        |

## Notes

- InfoGAN and VAE-GAN require more care during training due to additional losses.
- StyleGAN uses AdaIN layers for style modulation, improving flexibility.
- FID and KID scores computed on 1,000 samples from each model.
- Inception Scores remain low due to dataset complexity and size.

## License

MIT License
