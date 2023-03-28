from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout

class MyBoxLayout(GridLayout):
    def __init__(self, **kwargs):
        super(GridLayout, self).__init__(**kwargs)
        self.cols = 3
        self.rows = 2
        # First row containing 3 buttons
        button1 = Button(text='Button1', background_color=[0.2, 1, 0.2, 1], size_hint=(200, 100),pos=(100,500))
        button2 = Button(text='Button2', background_color=[0.2, 1, 0.2, 1], size_hint=(200, 100),pos=(300,500))
        button3 = Button(text='Button3', background_color=[0.2, 1, 0.2, 1], size_hint=(200, 100),pos=(500,500))
        self.add_widget(button1)
        self.add_widget(button2)
        self.add_widget(button3)


        # Second row containing two boxes
       
        
        box2a = BoxLayout(orientation='vertical', size_hint=(2, 1),pos=(200,400))
        textinput1 = TextInput(text='', multiline=False,size_hint=(1,2))
        button4 = Button(text='Go')
        button4.bind(on_press=lambda x: self.display_list(textinput1))
        self.label1 = Label(text="", font_size=20)
        self.label2 = Label(text='', font_size=20)
        self.label3 = Label(text='', font_size=20)
        box2a.add_widget(textinput1)
        box2a.add_widget(button4)
        box2a.add_widget(self.label1)
        box2a.add_widget(self.label2)
        box2a.add_widget(self.label3)
        self.add_widget(box2a)

        box2b = BoxLayout(orientation='vertical', size_hint=(0.5, 1),pos=(400,400))
        textinput2 = TextInput(text='', multiline=False,size_hint=(1,2))
        textinput3 = TextInput(text='', multiline=False,size_hint=(1,2))
        button5 = Button(text='Go')
        button5.bind(on_press=lambda x: self.display_text(textinput2, textinput3))
        self.label4 = Label(text='', font_size=20,size_hint=(1,2))
        box2b.add_widget(textinput2)
        box2b.add_widget(textinput3)
        box2b.add_widget(button5)
        box2b.add_widget(self.label4)

        self.add_widget(box2b)
    def display_list(self, textinput):
        self.label1.text = textinput.text
        self.label2.text = textinput.text
        self.label3.text = textinput.text

    def display_text(self, textinput1, textinput2):
        self.label4.text = textinput1.text + ' ' + textinput2.text


class MyApp(App):
    def build(self):
        return MyBoxLayout()


if __name__ == '__main__':
    MyApp().run()
