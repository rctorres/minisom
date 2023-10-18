
echo "Creating lpsds installation packages"
python -m build -o /tmp/dist

echo "Submitting newly packages to pypi"
python -m twine upload /tmp/dist/*

echo "Submission completed!"
