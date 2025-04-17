from nicegui import ui, app

class NestingGUI:
    def __init__(self):
        # Set default container dimensions in pixels
        self.container_width = 800
        self.container_height = 600
        self.path_static_img = '/img'
        app.add_static_files(self.path_static_img, './assets/img')
        self.setup_layout()

    def setup_layout(self):
        with ui.column().classes('w-full h-screen items-center justify-center p-0 m-0 gap-0'):
            # The canvas is an element whose dimensions we control via style.
            with ui.element('div').classes('relative').style(
                f'width: {self.container_width}px; height: {self.container_height}px;'
            ) as self.canvas:
                # Millimiter paper background
                ui.image(f'{self.path_static_img}/millimiter_paper_1500_900.png') \
                    .classes('absolute top-0 left-0 w-full h-full object-contain')
            # Bottom bar container for controls
            with ui.row().classes('w-full h-[10%] items-center justify-center gap-4'):
                # Input fields for container dimensions
                self.width_input = ui.number(
                    label='Container Width (px)',
                    value=self.container_width,
                    on_change=self.update_canvas_dimensions
                )
                self.height_input = ui.number(
                    label='Container Height (px)',
                    value=self.container_height,
                    on_change=self.update_canvas_dimensions
                )
                ui.button('Reset', on_click=self.reset_position)

    def update_canvas_dimensions(self, e):
        # Update container dimensions from the inputs—if a value is provided.
        self.container_width = self.width_input.value or self.container_width
        self.container_height = self.height_input.value or self.container_height
        
        # Update the canvas style with the new dimensions.
        self.canvas.style(f'width: {self.container_width}px; height: {self.container_height}px;')

    def reset_position(self):
                self.circle.style('left: 150px; top: 150px;')


NestingGUI()
ui.run(port=8080)
