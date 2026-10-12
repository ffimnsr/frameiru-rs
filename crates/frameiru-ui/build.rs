fn main() {
    slint_build::compile("ui/main.slint").expect("compile ui/main.slint");
    slint_build::compile("ui/tray.slint").expect("compile ui/tray.slint");
}
