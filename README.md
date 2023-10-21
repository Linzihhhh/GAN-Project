# GAN-Project
GAN First Project
This is a project for generating a segment of piano music about 10 sec using WGAN.

*Volume Warning: Please avoid using headsets for the following audio files*

GANv0 A GAN with simple ANN for generator and discriminator
results:

0th generation, 100 steps: 

https://user-images.githubusercontent.com/61591276/216775089-2c745ee3-fbb2-49e5-80c4-6a4a4286e5af.mp4

Obviously, no sound appear.

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

1th generation 200 step:

https://github.com/Linzihhhh/GAN-Project/blob/main/GANv1%20result/1th%20200step.mp3

Clear piano sound with rythm , but noise is loud

20th generation 3600 step:

https://github.com/Linzihhhh/GAN-Project/blob/main/GANv1%20result/20th%203600step.mp3

Several clips of piano appears, but noise still appear

49th generation 8600 step:

https://github.com/Linzihhhh/GAN-Project/blob/main/GANv1%20result/49th%208600step.mp3

The result is same as 20th, the reason may be overfitting or inadequate epochs, whatever we stop training.










