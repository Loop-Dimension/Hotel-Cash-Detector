# Generated migration for polygon zones and Gemini settings

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cctv', '0003_camera_cash_drawer_zone_enabled_and_more'),
    ]

    operations = [
        # Add polygon zone support to Camera model
        migrations.AddField(
            model_name='camera',
            name='cashier_zone_polygon',
            field=models.TextField(blank=True, null=True, help_text='JSON array of polygon points [[x1,y1],[x2,y2],...] for cashier zone'),
        ),
        migrations.AddField(
            model_name='camera',
            name='cash_drawer_zone_polygon',
            field=models.TextField(blank=True, null=True, help_text='JSON array of polygon points [[x1,y1],[x2,y2],...] for cash drawer zone'),
        ),
        migrations.AddField(
            model_name='camera',
            name='use_polygon_zones',
            field=models.BooleanField(default=False, help_text='Use polygon zones instead of rectangular zones'),
        ),
        
        # Add Gemini prompt customization fields
        migrations.AddField(
            model_name='camera',
            name='gemini_cash_prompt',
            field=models.TextField(blank=True, null=True, help_text='Custom Gemini prompt for cash detection validation'),
        ),
        migrations.AddField(
            model_name='camera',
            name='gemini_violence_prompt',
            field=models.TextField(blank=True, null=True, help_text='Custom Gemini prompt for violence detection validation'),
        ),
        migrations.AddField(
            model_name='camera',
            name='gemini_fire_prompt',
            field=models.TextField(blank=True, null=True, help_text='Custom Gemini prompt for fire detection validation'),
        ),
        
        # Add Gemini logging model
        migrations.CreateModel(
            name='GeminiLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('event_type', models.CharField(max_length=20)),
                ('is_validated', models.BooleanField(default=False)),
                ('confidence', models.FloatField(default=0.0)),
                ('reason', models.TextField(blank=True)),
                ('prompt_used', models.TextField(blank=True)),
                ('response_raw', models.TextField(blank=True, help_text='Raw JSON response from Gemini')),
                ('image_path', models.CharField(blank=True, max_length=500, null=True)),
                ('processing_time_ms', models.IntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('camera', models.ForeignKey(on_delete=models.deletion.CASCADE, related_name='gemini_logs', to='cctv.camera')),
            ],
            options={
                'db_table': 'gemini_logs',
                'ordering': ['-created_at'],
            },
        ),
    ]
