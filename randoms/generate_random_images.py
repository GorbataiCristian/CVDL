import requests


name = "random"


def request_image(imageId):
    response = requests.get("https://picsum.photos/400")

    with open(f'dataset_random\\{name}_{imageId}.png', 'wb') as file:
        file.write(response.content)

    print(f'Done: {imageId}')


def main():
    for i in range(1000):
        request_image(i)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

