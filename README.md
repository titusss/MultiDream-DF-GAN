# Generate dream visualizations using DF-GAN

Heavily based on [DF-GAN: Deep Fusion Generative Adversarial Networks for Text-to-Image Synthesis](https://github.com/tobran/DF-GAN).

## Prerequisites

- [Docker](https://www.docker.com/)

## How to run the web server

Install Docker image (do this one time, or every time you change the source code)
```
cd server
docker build -t dream-df-gan .
```

Run:
```
docker run -it -p [PORT]:5000 dream-df-gan
```

To run in the background:
```
docker run -d -it -p [PORT]:5000 dream-df-gan
```

## How to generate dream visualizations

```
http://localhost:[PORT]/[ENTER YOUR DREAM HERE]
```

## How to deploy to AWS

1. Install [EB CLI](https://github.com/aws/aws-elastic-beanstalk-cli-setup)

2. When setting the app for the first time, `cd aws-eb/` then run `eb init` and `eb create` and follow the instructions.

3. I'm using DockerHub to build the image locally and let AWS download it (building it takes too much memory for the t2.micro's). The image is configured in `aws-ebDockerrun.aws.json`. You can build your own image by running
```
docker build -t docker-username/dream-df-gan:latest .
docker push docker-username/dream-df-gan:latest
```
And change `Dockerrun.aws.json` to have the right image name.

Every change in the code you make, rebuild the image and push using the above commands.

5. To deploy your changes to aws, go to `cd aws-eb/` and run `eb deploy`

6. Configure the AWS Elastic Beanstalk as you wish using the AWS console. The default is using t2.micro machine, minimum 1 maximum 4. A single t2.micro machine is under the Free Tier, multiple are not.
