# Generated manually for adding validation_type and video_path to GeminiLog

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cctv', '0009_add_gemini_prompts_model'),
    ]

    operations = [
        migrations.AddField(
            model_name='geminilog',
            name='validation_type',
            field=models.CharField(
                choices=[('image', 'Image (Screenshot)'), ('video', 'Video (3 sec clip)')],
                default='image',
                help_text='Whether validation was done via image or video',
                max_length=10
            ),
        ),
        migrations.AddField(
            model_name='geminilog',
            name='video_path',
            field=models.CharField(
                blank=True,
                help_text='Path to validation video clip if used',
                max_length=500,
                null=True
            ),
        ),
    ]
