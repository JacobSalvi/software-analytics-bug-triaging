import npyscreen

from src.Database import Database
from src.model.Predictor import Predictor


class BugTriagingApp(npyscreen.NPSAppManaged):
    def onStart(self):
        self.addForm('MAIN', MainForm)
        self.addForm('ERROR', ErrorForm)

class MainForm(npyscreen.Form):
    def __init__(self, *args, **keywords):
        super(MainForm, self).__init__(*args, **keywords)
        self._predictor = Predictor()

    def create(self):
        self.add(npyscreen.TitleText, name='BugTriaging', editable=False)
        self.input_field = self.add(npyscreen.TitleText, name='Insert an issue number:', value='')
        self.input_field.when_value_edited = self.limit_to_numbers
        self.results_display = self.add(npyscreen.MultiLineEdit, value='', editable=False, max_height=10)

        self.submit_button = self.add(npyscreen.ButtonPress, name='Submit', when_pressed_function=self.display_results)

        self.clean_button = self.add(npyscreen.ButtonPress, name='Clean', when_pressed_function=self.clean_fields)
        self.exit_button = self.add(npyscreen.ButtonPress, name='Exit', when_pressed_function=self.exit_app)

    def limit_to_numbers(self):
        current_value = self.input_field.value
        self.input_field.value = ''.join(filter(str.isdigit, current_value))
        self.input_field.display()

    def exit_app(self):
        self.parentApp.switchForm(None)
        

    def display_results(self):
        user_input = self.input_field.value
        try:
            candidates = self._predictor.predict_assignees_by_issue_number(int(user_input))
            users = [Database.get_user_by_id(el) for el in candidates]
            number_commits = [Database.get_commits_per_user(user.get("login")) for user in users]
            formatted_results = "\n".join([f"User: {el.get('login')} number of commits: {commits}"
                                           for el, commits in zip(users, number_commits)])
            self.results_display.value = formatted_results
            self.results_display.display()
        except Exception as e:
            error_form = self.parentApp.getForm('ERROR')
            error_form.set_error_message(str(e))
            self.parentApp.switchForm('ERROR')

    def clean_fields(self):
        self.input_field.value = ''
        self.results_display.value = ''
        self.results_display.display()

class ErrorForm(npyscreen.Form):
    def create(self):
        self.error_message = self.add(npyscreen.TitleText, name='Error:', value='', editable=False)
        self.back_button = self.add(npyscreen.ButtonPress, name='Back', when_pressed_function=self.go_back)
        self.exit_button = self.add(npyscreen.ButtonPress, name='Exit', when_pressed_function=self.exit_app)

    def set_error_message(self, message):
        self.error_message.value = message
        self.error_message.display()

    def go_back(self):
        self.parentApp.switchForm('MAIN')

    def exit_app(self):
        self.parentApp.switchForm(None)

def main():
    app = BugTriagingApp()
    app.run()

if __name__ == "__main__":
    main()
