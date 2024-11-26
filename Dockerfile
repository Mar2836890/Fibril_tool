FROM cdrx/pyinstaller-windows:python3

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Install any Python dependencies (if you have a requirements.txt file)
RUN pip install -r requirements.txt

# Build the executable using PyInstaller
RUN pyinstaller --onedir --add-data="Results;Results" --add-data="Images_to_test;Images_to_test" Tool.py
