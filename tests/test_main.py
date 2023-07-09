from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_detect_faces():
    response = client.get(
        "/faces/?image_url=https%3A%2F%2Fwww.ppic.org%2Fwp-content%2Fuploads%2FCrowd-of-Diverse-People_800x528-768x512.jpg", )
    assert response.status_code == 200
    assert response.json() == [
        {
            "x1": 403,
            "x2": 455,
            "y1": 0,
            "y2": 66
        },
        {
            "x1": 61,
            "x2": 104,
            "y1": 16,
            "y2": 79
        },
        {
            "x1": 365,
            "x2": 418,
            "y1": 64,
            "y2": 126
        },
        {
            "x1": 480,
            "x2": 532,
            "y1": 79,
            "y2": 149
        },
        {
            "x1": 92,
            "x2": 144,
            "y1": 111,
            "y2": 174
        },
        {
            "x1": 173,
            "x2": 241,
            "y1": 108,
            "y2": 191
        },
        {
            "x1": 608,
            "x2": 670,
            "y1": 97,
            "y2": 180
        },
        {
            "x1": 341,
            "x2": 407,
            "y1": 150,
            "y2": 235
        },
        {
            "x1": 526,
            "x2": 584,
            "y1": 157,
            "y2": 236
        },
        {
            "x1": 250,
            "x2": 319,
            "y1": 206,
            "y2": 299
        },
        {
            "x1": 415,
            "x2": 479,
            "y1": 229,
            "y2": 303
        },
        {
            "x1": 87,
            "x2": 155,
            "y1": 268,
            "y2": 349
        },
        {
            "x1": 574,
            "x2": 643,
            "y1": 272,
            "y2": 364
        },
        {
            "x1": 476,
            "x2": 551,
            "y1": 334,
            "y2": 433
        },
        {
            "x1": 539,
            "x2": 587,
            "y1": 22,
            "y2": 80
        },
        {
            "x1": 716,
            "x2": 768,
            "y1": 203,
            "y2": 302
        },
        {
            "x1": 639,
            "x2": 723,
            "y1": 393,
            "y2": 510
        },
        {
            "x1": 263,
            "x2": 315,
            "y1": 37,
            "y2": 106
        },
        {
            "x1": 330,
            "x2": 404,
            "y1": 390,
            "y2": 504
        },
        {
            "x1": 647,
            "x2": 705,
            "y1": 56,
            "y2": 129
        },
        {
            "x1": 143,
            "x2": 233,
            "y1": 396,
            "y2": 512
        },
        {
            "x1": 0,
            "x2": 50,
            "y1": 123,
            "y2": 222
        },
        {
            "x1": 182,
            "x2": 224,
            "y1": 0,
            "y2": 43
        },
        {
            "x1": 743,
            "x2": 768,
            "y1": 42,
            "y2": 111
        }
    ]


def test_enhance_face():
    response = client.post(
        "enhance/?image_url=https%3A%2F%2Fwww.ppic.org%2Fwp-content%2Fuploads%2FCrowd-of-Diverse-People_800x528-768x512.jpg",
        json={
            "x1": 285,
            "x2": 426,
            "y1": 373,
            "y2": 514
        },
    )
    assert response.status_code == 200


def test_frontalize_face():
    response = client.post(
        "frontalize/?image_url=https%3A%2F%2Fgithub.com%2FHRLTY%2FTP-GAN%2Fblob%2Fmaster%2Fdata-example%2F001_01_01_140_06_cropped.png%3Fraw%3Dtrue"
    )
    assert response.status_code == 200
