namespace Faysals_Bluetooth_Communication_Project
{
    partial class helpPage
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(helpPage));
            label1 = new Label();
            btnConnect = new Button();
            SuspendLayout();
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.BackColor = Color.Transparent;
            label1.FlatStyle = FlatStyle.Popup;
            label1.Font = new Font("Arial Rounded MT Bold", 12F, FontStyle.Regular, GraphicsUnit.Point, 0);
            label1.ForeColor = SystemColors.ButtonFace;
            label1.Location = new Point(107, 31);
            label1.Name = "label1";
            label1.Size = new Size(300, 198);
            label1.TabIndex = 2;
            label1.Text = resources.GetString("label1.Text");
            // 
            // btnConnect
            // 
            btnConnect.BackColor = Color.Transparent;
            btnConnect.Cursor = Cursors.Hand;
            btnConnect.FlatAppearance.BorderColor = Color.Green;
            btnConnect.FlatAppearance.BorderSize = 3;
            btnConnect.FlatAppearance.MouseDownBackColor = Color.FromArgb(0, 192, 0);
            btnConnect.FlatAppearance.MouseOverBackColor = Color.FromArgb(128, 255, 128);
            btnConnect.FlatStyle = FlatStyle.Popup;
            btnConnect.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point, 0);
            btnConnect.ForeColor = Color.Chartreuse;
            btnConnect.Location = new Point(229, 248);
            btnConnect.Name = "btnConnect";
            btnConnect.Size = new Size(60, 26);
            btnConnect.TabIndex = 4;
            btnConnect.Text = "OK?";
            btnConnect.UseVisualStyleBackColor = false;
            btnConnect.Click += btnConnect_Click;
            // 
            // helpPage
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackgroundImage = (Image)resources.GetObject("$this.BackgroundImage");
            BackgroundImageLayout = ImageLayout.Stretch;
            ClientSize = new Size(498, 404);
            Controls.Add(btnConnect);
            Controls.Add(label1);
            FormBorderStyle = FormBorderStyle.None;
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "helpPage";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "helpPage";
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Label label1;
        private Button btnConnect;
    }
}