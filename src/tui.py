import npyscreen


def some_fun(user_input):
    if user_input == "error":
        raise ValueError("An error occurred: Invalid input.")
    return [
        {"name": "Alice", "comment": user_input, "age": 30},
        {"name": "Bob", "comment": user_input, "age": 25},
        {"name": "Charlie", "comment": user_input, "age": 35},
    ]

class BugTriagingApp(npyscreen.NPSAppManaged):
    def onStart(self):
        self.addForm('MAIN', MainForm)
        self.addForm('ERROR', ErrorForm)

class MainForm(npyscreen.Form):
    def create(self):
        self.add(npyscreen.TitleText, name='BugTriaging', editable=False)
        self.input_field = self.add(npyscreen.TitleText, name='Input:', value='')
        self.results_display = self.add(npyscreen.MultiLineEdit, value='', editable=False, max_height=10)

        self.submit_button = self.add(npyscreen.ButtonPress, name='Submit', when_pressed_function=self.display_results)

        self.clean_button = self.add(npyscreen.ButtonPress, name='Clean', when_pressed_function=self.clean_fields)
        self.exit_button = self.add(npyscreen.ButtonPress, name='Exit', when_pressed_function=self.exit_app)

    def exit_app(self):
        self.parentApp.switchForm(None)
        

    def display_results(self):
        user_input = self.input_field.value
        try:
            results = some_fun(user_input)

            formatted_results = ""
            for item in results:
                formatted_results += f"Name: {item['name']}, Comment: {item['comment']}, Age: {item['age']}\n"

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
