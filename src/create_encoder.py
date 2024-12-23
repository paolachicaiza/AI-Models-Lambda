def create_encoder(data_values, data_encode):
    try:
        cols = list(data_values.keys())

        def encode_values(col, data_values, data_encode):
            try:
                to_values_encode = {}
                for i in range(len(data_values[col])):
                    to_values_encode[str(data_values[str(col)].iloc[i])] = data_encode[str(col)].iloc[i]
                return to_values_encode
            except Exception as e:
                print(e)
                print('Error')
                raise e

        map_encode = {}
        for col in cols:
            map_encode[str(col)] = encode_values(col, data_values, data_encode)
        return map_encode
    except Exception as error:
        print(error)
        print('Error')
        raise error
