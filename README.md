# GAN-Project
GAN First Project
This is a project for generating a segment of piano music about 10 sec using WGAN.

*Volume Warning: Please avoid using headsets for the following audio files*

GANv0 A GAN with simple ANN for generator and discriminator
results:

0th generation, 100 steps: 

https://user-images.githubusercontent.com/61591276/216775089-2c745ee3-fbb2-49e5-80c4-6a4a4286e5af.mp4

Strikingly, no sound appear.

100th generation, 10100 steps: 

https://user-images.githubusercontent.com/61591276/216775374-27752b98-becf-40e6-854c-b88b470d36f0.mp4

The some piano sound comes out, but along with copious amount of noise

300th generation, 30100 steps: 

https://user-images.githubusercontent.com/61591276/216775424-36fafe92-c59d-4bbb-bac6-208cd0691d66.mp4

We expect that noise may disappear during iterate, however the noise become bigger.

435th generation, 43600 steps: 

https://user-images.githubusercontent.com/61591276/216775460-ab7b7d6d-06f9-4b2f-ad65-1378d13f1f9c.mp4

Stop training, the noise is disruptive.



GANv1 Adjust model to RNN for generator and discriminator
result:



