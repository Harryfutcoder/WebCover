# WebRLED

## 0. What is WebRLED

![WebRLED_Overview](./assets/overview.png)
WebRLED is a novel approach to testing automatic web applications based on deep reinforcement learning, 
which has the following characteristics: 

1) a grid-based action value learning and estimation method combining DQN (Deep Q-Network) and upsampling technique, which can dramatically improve the efficiency of exploration;
2) a novel action discriminator trained during the exploration, which can identify more actionable web element; 
3) an adaptive, curiosity-driven reward model, which considers the novelty of an explored state within an episode and global history, and can guide the agent effectively to explore more diverse states.

## 1. Install WebRLED via Zip file

We have packaged the complete WebRLED into a ZIP file `webrled.zip`, which can be run simply by configuring the Python environment. 

### 1.1 How to use?

User need to follow the steps below to use WebRLED:

#### Step 1: Installation

Create Conda virtual environment & install python (3.10) dependencies

```bash
unzip webrled.zip
cd webrled
conda create -n webrled python=3.10
conda activate webrled
pip install -r requirements.txt
playwright install
export PYTHONPATH=path/to/webrled
```

path/to/webrled: Replace this with the absolute address of the webrled folder

#### Step 2: Using an online application as an example

We use an online application called Splittypie as an example.
This application is online and does not require deployment. It is one of the six applications in our first benchmark.

**Simply execute the following command to start** ( please make sure the Python Conda environment is functioning properly in the first step): `python src/main.py --appname splittypie --url https://splittypie.com/ --domain splittypie.com`. 

However, since it is online, coverage information cannot be obtained. To obtain coverage information, please refer to Step [3](#step-3-deploy-the-tested-web-applications-and-launch-experiments) for deployment and testing.

#### Step 3 Deploy the tested web applications and Launch Experiments

##### 3.1 First benchmark

1. Deploy 6 applications on local machine.

The following six web applications (dimeshift, pagekit, splittypie, phoenix-trello, retroboard, petclinic) are packaged into Docker images by the authors of DIG and are used as benchmark tests by WebExplor. Detailed packaging instructions can be found in the [documentation](https://github.com/matteobiagiola/FSE19-submission-material-DIG?tab=readme-ov-file#112-clone-repo-and-download-docker-images).

For ease of use, the Docker images for these applications are stored in the following location: `./webapps/first_benchmark`. To deploy them on the local machine, please follow these steps:

Import each application into Docker:
- Dimeshift: `docker load < ./webapps/first_benchmark/dimeshift.tar`
- Pagekit: `docker load < ./webapps/first_benchmark/pagekit.tar`
- Splittypie: `docker load < ./webapps/first_benchmark/splittypie.tar`
- Phoenix: `docker load < ./webapps/first_benchmark/phoenix.tar`
- Retroboard: `docker load < ./webapps/first_benchmark/retroboard.tar`
- Petclinic: `docker load < ./webapps/first_benchmark/petclinic.tar`

Run each application (Note: Test one application at a time. Only run one application per test):
- Dimeshift: `docker run -it --workdir=/home/dimeshift-application --name=dimeshift --expose 8080 --expose 3306 -p 4000:8080 -p 3306:3306 -d --entrypoint ./run-code-instrumentation.sh dockercontainervm/dimeshift:latest bash`
- Pagekit: `docker run -it --workdir=/var/www/html/pagekit --name=pagekit --expose 80 --expose 3306 -p 4001:80 -p 3306:3306 --entrypoint ./run-code-instrumentation.sh -d dockercontainervm/pagekit:latest bash`
- Splittypie: `docker run -it --workdir=/home/splittypie --name=splittypie --expose 4200 -p 4005:4200 -d --entrypoint ./run-code-instrumentation.sh dockercontainervm/splittypie:latest bash`
- Phoenix-trello: `docker run -it --workdir=/home/phoenix-trello --name=phoenix --expose 4000 --expose 5432 -p 4003:4000 -p 5432:5432 --env PATH=/root/.kiex/elixirs/elixir-1.3.1/bin:/root/.kiex/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin -d --entrypoint ./run-code-instrumentation.sh dockercontainervm/phoenix-trello:latest bash`
- Retroboard: `docker run -it --workdir=/home/retro-board --name=retroboard --expose 8080 -p 4004:8080 -d --entrypoint ./run-code-instrumentation.sh dockercontainervm/retroboard:latest bash`
- Petclinic: `docker run -it --workdir=/home/spring-petclinic-angularjs --name=petclinic --expose 8080 --expose 3306 -p 4002:8080 -p 3306:3306 -d --env PATH=/root/workspace/maven/apache-maven-3.5.4/bin:/root/workspace/java/jdk1.8.0_181/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --entrypoint ./run-code-instrumentation.sh dockercontainervm/petclinic:latest bash`

2. Run express-istanbul to gather coverage information.

Node version: 10.24.1

```bash
unzip express-istanbul.zip
cd express-istanbul
npm install
node app.js
```

3. Use WebRLED for testing.

To test the application and gather coverage, please run the following command:

```bash
python src/main.py --appname <web application name> --url <URL of the app> --domain <the domain corresponding to the URL> --coverage
```

For example, `python src/main.py --appname splittypie --url http://localhost:4200/ --domain http://localhost:4200/ --coverage`. The `<web application name>` should be the same as the one prompted for help, otherwise it can't reset and get coverage. Help can be obtained with the command `python src/main.py --help`.

Additionally, we have configured the URLs and domains for these 12 applications in the code for local deployment. Simply enter the appname in the run command:
```bash
python src/main.py --appname <web application name> --coverage
```
For example, `python src/main.py --appname splittypie --coverage`.


##### 3.2 Second benchmark

1. Deploy 6 applications on local machine.

The following six web applications (realworld,timeoff,parabank,agilefant,4gaboards,gadael) are collected from recent work and Github. For ease of use, the Docker images for these applications are stored in the following location: `./webapps/second_benchmark`. To deploy them on the local machine, please follow these steps (Note: Realworld and 4gaBoards):

Import each application into Docker:
- Timeoff: `docker load < ./webapps/second_benchmark/timeoff.tar`
- Parabank: `docker load < ./webapps/second_benchmark/parabank.tar`
- Agilefant: `docker load < ./webapps/second_benchmark/agilefant.tar`
- Gadael: `docker load < ./webapps/second_benchmark/gadael.tar`

Note: After deploying Realworld and 4gaBoards as Docker containers, exceptions occur during runtime. We have provided corresponding compressed packages that contain readme documentation with deployment instructions.

Run each application (Note: Test one application at a time. Only run one application per test):
- Timeoff: `docker run -it --name timeoff-flask-container -p 6969:6969 -p 3000:3000 -d timeoff`
- Parabank: `docker run -it --name parabank-flask-container -p 6969:6969 -p 8080:8080 -d parabank`
- Agilefant: `docker run -it --name agilefant-flask-container -p 6969:6969 -p 8080:8080 -p 3306:3306 -d agilefant`
- Gadael: `docker run -it --name gadael-flask-container -p 6969:6969 -p 3000:3000 -d gadael`

2. Use WebRLED for testing.

To test the application and gather coverage, please run the following command:

```bash
python src/main.py --appname <web application name> --url <URL of the app> --domain <the domain corresponding to the URL> --coverage
```

For example, `python src/main.py --appname timeoff --url http://localhost:3000/ --domain http://localhost:3000 --coverage`. The `<web application name>` should be the same as the one prompted for help, otherwise it can't reset and get coverage. Help can be obtained with the command `python src/main.py --help`.

Additionally, we have configured the URLs and domains for these 12 applications in the code for local deployment. Simply enter the appname in the run command:
```bash
python src/main.py --appname <web application name> --coverage
```
For example, `python src/main.py --appname timeoff --coverage`.

##### 3.3 Third benchmark

###### real-world applications

Here we have selected top 50 real applications to test WebRLED's ability to find failures by using the command `python src/main.py --appname real --url <web application url>. For example, python src/main.py --appname real --url 'https://www.quora.com/'`.

Since we are testing randomly, the coverage is not 100% and there may be some special circumstances that make the program go wrong. We verified the test on ubuntu 22.04.

