"""
Django management command to seed initial data for the CCTV system.
Creates sample regions, branches, cameras, and an admin user.
"""

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from cctv.models import Region, Branch, Camera, Event, BranchAccount
from django.utils import timezone
import random

User = get_user_model()


class Command(BaseCommand):
    help = 'Seed initial data for the CCTV system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before seeding',
        )

    def handle(self, *args, **options):
        if options['clear']:
            self.stdout.write('Clearing existing data...')
            Event.objects.all().delete()
            Camera.objects.all().delete()
            BranchAccount.objects.all().delete()
            Branch.objects.all().delete()
            Region.objects.all().delete()
            User.objects.filter(is_superuser=False).delete()

        self.stdout.write('Seeding data...')

        # Create admin user if not exists
        if not User.objects.filter(username='admin').exists():
            admin = User.objects.create_superuser(
                username='admin',
                email='admin@hotel.com',
                password='admin123',
                first_name='관리자',
                role='admin'
            )
            self.stdout.write(self.style.SUCCESS(f'Created admin user: admin / admin123'))
        else:
            admin = User.objects.get(username='admin')
            self.stdout.write('Admin user already exists')

        # Create regions
        regions_data = [
            {'name': '서울', 'code': 'SEL'},
            {'name': '경기', 'code': 'GGI'},
            {'name': '부산', 'code': 'BSN'},
            {'name': '대구', 'code': 'DGU'},
            {'name': '인천', 'code': 'ICN'},
        ]

        regions = []
        for region_data in regions_data:
            region, created = Region.objects.get_or_create(
                code=region_data['code'],
                defaults={'name': region_data['name']}
            )
            regions.append(region)
            if created:
                self.stdout.write(f'Created region: {region.name}')

        # Create branches
        branches_data = [
            {'name': '강남점', 'region': 0, 'address': '서울시 강남구 역삼동 123'},
            {'name': '홍대점', 'region': 0, 'address': '서울시 마포구 서교동 456'},
            {'name': '명동점', 'region': 0, 'address': '서울시 중구 명동 789'},
            {'name': '판교점', 'region': 1, 'address': '경기도 성남시 분당구 판교로 100'},
            {'name': '일산점', 'region': 1, 'address': '경기도 고양시 일산동구 200'},
            {'name': '해운대점', 'region': 2, 'address': '부산시 해운대구 해운대해변로 300'},
            {'name': '서면점', 'region': 2, 'address': '부산시 부산진구 서면로 400'},
            {'name': '동성로점', 'region': 3, 'address': '대구시 중구 동성로 500'},
            {'name': '송도점', 'region': 4, 'address': '인천시 연수구 송도동 600'},
        ]

        branches = []
        for branch_data in branches_data:
            branch, created = Branch.objects.get_or_create(
                name=branch_data['name'],
                defaults={
                    'region': regions[branch_data['region']],
                    'address': branch_data['address'],
                    'status': random.choice(['confirmed', 'reviewing', 'pending'])
                }
            )
            branches.append(branch)
            if created:
                self.stdout.write(f'Created branch: {branch.name}')

        # Create project managers
        pm_data = [
            {'username': 'pm_seoul', 'name': '김서울', 'branches': [0, 1, 2]},
            {'username': 'pm_gyeonggi', 'name': '이경기', 'branches': [3, 4]},
            {'username': 'pm_busan', 'name': '박부산', 'branches': [5, 6]},
        ]

        for pm in pm_data:
            if not User.objects.filter(username=pm['username']).exists():
                user = User.objects.create_user(
                    username=pm['username'],
                    email=f'{pm["username"]}@hotel.com',
                    password='pm123',
                    first_name=pm['name'],
                    role='project_manager'
                )
                # Assign branches using managers relation on Branch
                for branch_idx in pm['branches']:
                    branches[branch_idx].managers.add(user)
                self.stdout.write(self.style.SUCCESS(f'Created PM user: {pm["username"]} / pm123'))

        # Create cameras
        camera_locations = ['캐셔', '로비', '입구', '주차장', '복도', '엘리베이터', '카운터']
        
        for branch in branches:
            num_cameras = random.randint(2, 4)
            for i in range(num_cameras):
                camera_name = random.choice(camera_locations)
                camera_id = f'CAM-{branch.region.code}-{str(i+1).zfill(2)}'
                camera, created = Camera.objects.get_or_create(
                    branch=branch,
                    camera_id=camera_id,
                    defaults={
                        'name': f'{camera_name} 카메라 {i+1}',
                        'location': camera_name,
                        'rtsp_url': f'rtsp://192.168.1.{random.randint(100, 200)}:554/stream{i+1}',
                        'status': random.choice(['online', 'online', 'online', 'offline']),  # 75% online
                        'detect_cash': random.choice([True, False]),
                        'detect_violence': random.choice([True, False]),
                        'detect_fire': random.choice([True, False]),
                    }
                )
                if created:
                    self.stdout.write(f'Created camera: {branch.name} - {camera.name}')

                    # Create some events for this camera
                    num_events = random.randint(0, 5)
                    for j in range(num_events):
                        event_type = random.choice(['cash', 'violence', 'fire'])
                        hours_ago = random.randint(1, 72)
                        
                        Event.objects.create(
                            branch=branch,
                            camera=camera,
                            event_type=event_type,
                            confidence=random.uniform(0.7, 0.99),
                            status=random.choice(['confirmed', 'pending', 'reviewing']),
                            frame_number=random.randint(1000, 50000),
                            bbox_x1=random.randint(0, 200),
                            bbox_y1=random.randint(0, 200),
                            bbox_x2=random.randint(300, 600),
                            bbox_y2=random.randint(300, 400),
                        )

        # Create branch accounts
        account_names = ['김지점장', '이매니저', '박스태프', '최관제']
        for idx, branch in enumerate(branches[:4]):  # First 4 branches get accounts
            BranchAccount.objects.get_or_create(
                branch=branch,
                email=f'staff{idx}@hotel.com',
                defaults={
                    'name': account_names[idx % len(account_names)],
                    'role': random.choice(['manager', 'staff', 'control'])
                }
            )
            self.stdout.write(f'Created branch account for: {branch.name}')

        # Summary
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('=' * 50))
        self.stdout.write(self.style.SUCCESS('Data seeding completed!'))
        self.stdout.write(self.style.SUCCESS('=' * 50))
        self.stdout.write(f'Regions: {Region.objects.count()}')
        self.stdout.write(f'Branches: {Branch.objects.count()}')
        self.stdout.write(f'Cameras: {Camera.objects.count()}')
        self.stdout.write(f'Events: {Event.objects.count()}')
        self.stdout.write(f'Users: {User.objects.count()}')
        self.stdout.write('')
        self.stdout.write('Login credentials:')
        self.stdout.write('  Admin: admin / admin123')
        self.stdout.write('  Project Manager: pm_seoul / pm123')
        self.stdout.write('  Project Manager: pm_gyeonggi / pm123')
        self.stdout.write('  Project Manager: pm_busan / pm123')
