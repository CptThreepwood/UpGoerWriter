import os
import io
import sys
import time
import shutil
import zipfile
import pathlib
import subprocess


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
BUILD_DIR = os.path.join(ROOT_DIR, 'build')
PACKAGE_DIR = os.path.join(BUILD_DIR, 'python')
PACKAGE_REQ = os.path.join(ROOT_DIR, 'requirements_slimmed.txt')


def changed_requirements():
    current_packages = subprocess.run([
        sys.executable, "-m", "pip", "freeze",
        '--path', PACKAGE_DIR,
    ], capture_output=True, text=True)

    pkg_io = open(PACKAGE_REQ, 'r')
    test = pkg_io.read()
    return test != current_packages.stdout


def install_packages():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        '-r', PACKAGE_REQ,
        '--target', PACKAGE_DIR,
    ])


def create_package_layer():
    ## Install runtime packages
    if changed_requirements():
        install_packages()

        with zipfile.ZipFile(os.path.join(BUILD_DIR, 'packages.zip'), 'w') as package_zip:
            package_files = [
                pathlib.Path(os.path.join(dp, f))
                for dp, dn, fn in os.walk(PACKAGE_DIR)
                for f in fn
            ]
            for f in package_files:
                package_zip.write(f, f.relative_to(BUILD_DIR))



def generate_deployment_name():
    return os.path.join(BUILD_DIR, 'UpGoerWriter_{}'.format(round(time.time())))


def copy_source(deployment_dir):
    shutil.copytree(os.path.join(ROOT_DIR, 'Lexica'), os.path.join(deployment_dir, 'Lexica'))
    shutil.copytree(os.path.join(ROOT_DIR, 'SpacyTranslation'), os.path.join(deployment_dir, 'SpacyTranslation'))
    shutil.copyfile(os.path.join(ROOT_DIR, 'lambdaHandler', 'index.py'), os.path.join(deployment_dir, 'index.py'))


def create_deployment():
    ## Get a unique name for the deployment
    deployment_dir = generate_deployment_name()

    ## Make the directory for the deployment
    if not os.path.exists(deployment_dir):
        os.makedirs(deployment_dir)

    ## Copy in source files
    copy_source(deployment_dir)

    archive_name = '{0}.zip'.format(os.path.basename(deployment_dir))
    with zipfile.ZipFile(os.path.join(BUILD_DIR, archive_name) , 'w') as package_zip:
        package_files = [
            pathlib.Path(os.path.join(dp, f))
            for dp, dn, fn in os.walk(deployment_dir)
            for f in fn
        ]
        for f in package_files:
            package_zip.write(f, f.relative_to(deployment_dir))


if __name__ == '__main__':
    create_deployment()
    create_package_layer()