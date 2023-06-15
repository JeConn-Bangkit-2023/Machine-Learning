FROM tensorflow/serving

WORKDIR /nsfw_model

COPY . /nsfw_model

EXPOSE 8605

ENTRYPOINT ["tensorflow_model_server", "--rest_api_port=8605", "--model_name=nsfw_model", "--model_base_path=/nsfw_model/saved_model/"]
