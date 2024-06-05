# cifar10_app/views.py
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.conf import settings
from .models import Image
from tensorflow_model.model import predict, class_names

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        image_path = default_storage.save(image_file.name, image_file)
        image_full_path = str(settings.MEDIA_ROOT / image_path)

        # Predict the class of the image
        predicted_class, prediction_confidences = predict(image_full_path)

        # Save to the database
        image = Image(image=image_path, prediction=predicted_class)
        image.save()

        # Store prediction confidences in session
        request.session['prediction_confidences'] = prediction_confidences

        return redirect('result', image_id=image.id)
    return render(request, 'upload.html')

def result(request, image_id):
    image = Image.objects.get(id=image_id)
    prediction_confidences = request.session.get('prediction_confidences')
    return render(request, 'result.html', {
        'image': image,
        'class_names': class_names,
        'prediction_confidences': prediction_confidences
    })
