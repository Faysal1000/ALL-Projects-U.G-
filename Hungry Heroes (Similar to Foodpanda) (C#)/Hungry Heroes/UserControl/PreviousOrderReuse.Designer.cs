namespace Hungry_Heroes
{
    partial class PreviousOrderReuse
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            lresname = new Label();
            display = new DataGridView();
            ((System.ComponentModel.ISupportInitialize)display).BeginInit();
            SuspendLayout();
            // 
            // lresname
            // 
            lresname.Anchor = AnchorStyles.None;
            lresname.AutoSize = true;
            lresname.BackColor = Color.Transparent;
            lresname.Font = new Font("Arial Rounded MT Bold", 14.25F, FontStyle.Underline, GraphicsUnit.Point);
            lresname.Location = new Point(54, 40);
            lresname.Name = "lresname";
            lresname.Size = new Size(170, 22);
            lresname.TabIndex = 15;
            lresname.Text = "Previous Orders--";
            // 
            // display
            // 
            display.AllowUserToAddRows = false;
            display.AllowUserToDeleteRows = false;
            display.Anchor = AnchorStyles.None;
            display.BackgroundColor = SystemColors.ActiveBorder;
            display.BorderStyle = BorderStyle.Fixed3D;
            display.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            display.Location = new Point(54, 65);
            display.Name = "display";
            display.ReadOnly = true;
            display.RowTemplate.Height = 25;
            display.Size = new Size(641, 349);
            display.TabIndex = 16;
            // 
            // PreviousOrderReuse
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            Controls.Add(display);
            Controls.Add(lresname);
            Name = "PreviousOrderReuse";
            Size = new Size(745, 476);
            Load += PreviousOrderReuse_Load;
            ((System.ComponentModel.ISupportInitialize)display).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion
        private Label lresname;
        private DataGridView display;
    }
}
