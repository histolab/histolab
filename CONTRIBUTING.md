# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change. 

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Contribution guidelines and standards
Before sending your PR for review, make sure your changes are consistent with the guidelines and follow the coding 
style.

### General guidelines and philosophy for contribution
- Include unit tests when you contribute new features, as they help to a) prove that your code works correctly, and b) 
  guard against future breaking changes to lower the maintenance cost.
- Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient test 
  coverage.
- Keep API compatibility in mind when you change code in Histolab core.
- Tests coverage cannot decrease from the current %.
- Do not push integration tests without unit tests.

## Contribution Workflow

Code contributions—bug fixes, new development, test improvement—all follow a GitHub-centered workflow. To participate 
in Histolab development, set up a GitHub account. Then:

 1. Fork the repo https://github.com/histolab/histolab. Go to the project repo page and use the Fork button. This will 
 create a copy of the repo, under your username. (For more details on how to fork a repository see 
 [this guide](https://help.github.com/articles/fork-a-repo/).)

 2. Clone down the forked repo to your local machine. 
   
    `$ git clone git@github.com:your-user-name/project-name.git`

 3. Create a new branch to hold your work.

    `$ git checkout -b new-branch-name`

 4. Work on your code. Write and run tests.

 5. Commit your changes.

    `$ git add .`
    
    `$ git commit -m "commit message here"`

 6. Push your changes to your GitHub repo.

    `$ git push origin branch-name`

 7. Open a Pull Request (PR). Go to the original project repo on GitHub. There will be a message about your recently 
    pushed branch, asking if you would like to open a pull request. Follow the prompts, compare across repositories, 
    and submit the PR. 
    For more read [here](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) 
 
 8. Maintainers and other contributors will review your PR. Please participate in the conversation, 
    and try to make any requested changes. Once the PR is approved, the code will be merged.

Before working on your next contribution, make sure your local repository is up to date.

 1. Set the upstream remote. (You only have to do this once per project, not every time.)

    `$ git remote add upstream git@github.com:histolab/histolab`

 2. Switch to the local master branch.

    `$ git checkout master`

 3. Pull down the changes from upstream.

    `$ git pull upstream master`

 4. Push the changes to your GitHub account. (Optional, but a good practice.)

     `$ git push origin master`

 5. Create a new branch if you are starting new work.

    `$ git checkout -b branch-name`

Additional git and GitHub resources:

- [Git documentation](https://git-scm.com/documentation)
- [Git development workflow](https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html)
- [Resolving merge conflicts](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/)

## Create your local environment

Before starting contributing to Histolab, test that your local environment is up and running. Here some steps:

- Create a python 3.6 - 3.7 `virtualenv`
- Activate the env and in the project root run:
  
  `pip install -e .[testing]`
  
  `pip install -r requirements-dev.txt`
  
- Install the pre-commit hooks (Optional, but useful for code style compliance)

   `pre-commit install` <- *to be ran in the project root directory*

- Run the tests
 
   `pytest`

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project a harassment-free experience for everyone, 
regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, 
personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information, such as a physical or electronic
  address, without explicit permission
- Other conduct which could reasonably be considered inappropriate in a
  professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community. Examples of
representing a project or community include using an official project e-mail
address, posting via an official social media account, or acting as an appointed
representative at an online or offline event. Representation of a project may be
further defined and clarified by project maintainers.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported by contacting one of the project mantainers/owners. All
complaints will be reviewed and investigated and will result in a response that
is deemed necessary and appropriate to the circumstances. The project team is
obligated to maintain confidentiality with regard to the reporter of an incident.
Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good
faith may face temporary or permanent repercussions as determined by other
members of the project's leadership.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4,
available at [http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/
