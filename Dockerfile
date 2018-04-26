### Using conda dockerfile template from:
# https://fmgdata.kinja.com/using-docker-with-conda-environments-1790901398

# Start from miniconda image:
FROM continuumio/miniconda


# Set the ENTRYPOINT to use bash
# (this is also where you’d set SHELL,
# if your version of docker supports this)
ENTRYPOINT ["/bin/bash", "-c"]

EXPOSE 5000


# Install some essential things:
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
 libpq-dev \
 build-essential \
 xvfb \
 vim \
&& rm -rf /var/lib/apt/lists/*


# Use the conda environment yaml file to create the "gctree" conda environment:
ADD environment.yml /tmp/environment.yml
WORKDIR /tmp
RUN ["conda", "env", "create", "-f", "environment.yml"]

# this doesn't work, have to activate the env by hand after running
# RUN ["/bin/bash", "-c", "source activate gctree"]

# Copy over the repository:
WORKDIR /gctree
COPY . /gctree
